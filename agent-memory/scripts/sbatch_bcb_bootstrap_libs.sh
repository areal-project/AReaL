#!/bin/bash
#SBATCH --job-name=yl-bcb-bootstrap
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=slurmd-23,slurmd-24,slurmd-45
#SBATCH --output=logs/bcb_bootstrap_%j.log
#SBATCH --error=logs/bcb_bootstrap_%j.log

# One-off job: populate /storage/.../bcb_eval_libs with BCB eval deps incl. TF/keras.
# Design:
#   - All installs use --no-deps to avoid pulling numpy 2.x (would clash with TF ABI).
#   - Explicitly install numpy<2 (1.26.x) + scipy compatible with numpy 1.26 + TF.
#   - The BCB eval subprocess in memrl/bigcodebench_eval/eval_utils.py will
#     sys.path.insert(0, TARGET_DIR) at worker entry, so this dir's libs are
#     only used inside the eval subprocess, not in the main vLLM/agent process.

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
TARGET_DIR="/storage/openpsi/users/yl/agent-memory/.cache/bcb_eval_libs"

echo "=========================================="
echo "BCB Eval Libs Bootstrap v2 (numpy<2 + TF)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Target: $TARGET_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Clean previous attempt to avoid numpy 2.x contamination.
if [ -d "${TARGET_DIR}" ]; then
    echo "[INFO] Removing old TARGET_DIR contents..."
    rm -rf "${TARGET_DIR}"
fi
mkdir -p "${TARGET_DIR}"

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
set -e
cd ${MEMRL_DIR}
echo '[INFO] Python: '\$(python --version)
echo '[INFO] Pip: '\$(pip --version)

echo ''
echo '=== Phase 1: install numpy 1.26 + scipy compatible + TF stack (--no-deps) ==='
# numpy must be <2 for TF 2.16-2.17 ABI compatibility.
# scipy 1.13.x is the latest that pairs with numpy 1.26.
pip install --target=${TARGET_DIR} --upgrade --no-deps \
    'numpy>=1.26,<2.0' \
    'scipy>=1.11,<1.14' \
    'h5py>=3.10' \
    'protobuf>=4.21,<5.30' \
    'grpcio>=1.59' \
    'ml-dtypes>=0.3.1,<0.5' \
    absl-py astunparse flatbuffers gast google-pasta libclang opt-einsum \
    tensorboard tensorflow-io-gcs-filesystem termcolor wrapt \
    optree namex rich packaging requests typing_extensions setuptools wheel six \
    2>&1 | tail -15

echo ''
echo '=== Phase 2: install tensorflow + tf-keras (legacy keras 2 API) ==='
pip install --target=${TARGET_DIR} --upgrade --no-deps \
    'tensorflow>=2.17,<2.18' 'tf-keras>=2.17,<2.18' 'keras>=3.4,<3.6' \
    2>&1 | tail -10

echo ''
echo '=== Phase 3: install BCB eval pure-Python libs (--no-deps, then deps for those with C ext) ==='
# Pure-Python libs first (--no-deps): these only need their own .py files.
pip install --target=${TARGET_DIR} --upgrade --no-deps \
    faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome \
    mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful \
    Levenshtein nltk Pillow matplotlib jinja2 werkzeug click itsdangerous markupsafe \
    blinker flask flask-restful flask-wtf wtforms \
    soundfile soxr audioread lazy-loader pooch decorator \
    rapidfuzz pyparsing cycler kiwisolver fonttools contourpy \
    et-xmlfile beautifulsoup4 soupsieve html5lib webencodings lxml cssselect \
    branca xyzservices pyproj pyogrio shapely \
    geographiclib certifi charset-normalizer idna urllib3 \
    smart_open msgpack regex tqdm threadpoolctl joblib \
    cffi pycparser cryptography \
    aniso8601 sqlparse asgiref \
    patsy python-dateutil pytz \
    'scikit-learn>=1.3,<1.6' 'python-http-client>=3.3' \
    2>&1 | tail -10

echo ''
echo '=== Phase 4: verify imports (with PYTHONPATH=TARGET_DIR) ==='
export PYTHONPATH=${TARGET_DIR}:\$PYTHONPATH
export TF_USE_LEGACY_KERAS=1
python <<'PY'
import sys, os
print('PYTHONPATH[0]:', sys.path[0])
print('TF_USE_LEGACY_KERAS=', os.environ.get('TF_USE_LEGACY_KERAS'))
# Check numpy version actually used
import numpy
print(f'numpy in use: {numpy.__version__}  (loaded from: {numpy.__file__})')

fails = []
mods = ['tensorflow','keras','tf_keras',
        'sklearn',
        'mechanize','wikipedia','wordcloud','wordninja',
        'requests_mock','Levenshtein','natsort','librosa','flask_wtf','flask_restful',
        'faker','statsmodels','django','openpyxl','docx','xlwt','sendgrid','skimage',
        'pyquery','geopandas','geopy','xmltodict','flask_mail','flask_login','pyfakefs',
        'texttable','textblob','gensim','pytesseract','holidays','folium','Cryptodome']
for mod in mods:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', '?')
        print(f'  OK   {mod:20s} {v}')
    except Exception as e:
        print(f'  FAIL {mod:20s} {type(e).__name__}: {str(e)[:100]}')
        fails.append(mod)

print()
if fails:
    print(f'IMPORT FAILURES ({len(fails)}): {fails}')
    sys.exit(1)
print('All imports OK.')
PY

echo ''
echo '=== Phase 5: sanity: build a tf.keras Sequential model (catches ABI issues at runtime) ==='
python <<'PY'
import os
import tensorflow as tf
print(f'tensorflow: {tf.__version__}')
print(f'tf.keras  : {tf.keras.__version__}')
import keras
print(f'keras (top-level): {keras.__version__}')

# Build + compile + predict — exercises numpy ABI and saved-checkpoint reader.
m = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
m.compile('sgd','mse')
import numpy as np
y = m.predict(np.zeros((2,3)), verbose=0)
print(f'tf.keras predict OK, output shape: {y.shape}')
PY

echo ''
echo '=== Phase 6: confirm a representative BCB task pattern works ==='
python <<'PY'
# Mimic what BCB task BCB/289 does: build a small TF model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
m = Sequential([Dense(8, activation='relu', input_shape=(4,)), Dense(1)])
m.compile(optimizer='adam', loss='mse')
print('BCB-style tf.keras task pattern: OK')
PY

echo ''
echo '=== Done. Sizes: ==='
du -sh ${TARGET_DIR}
df -h /storage | tail -2
"

EXIT=$?
echo ''
echo '=========================================='
echo "End time: $(date)"
echo "Exit code: $EXIT"
echo '=========================================='
