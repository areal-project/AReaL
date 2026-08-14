#!/usr/bin/env bash

set -o errexit

#pip install -r requirements.txt > /dev/null 2>&1
pip install \
mypy==1.14.1 \
mypy-extensions==1.0.0 \
pandas-stubs==2.2.3.241126 \
types-cffi==1.16.0.20241221 \
types-colorama==0.4.15.20240311 \
types-docker==7.1.0.20241229 \
types-docutils==0.21.0.20241128 \
types-openpyxl==3.1.5.20241225 \
types-Pillow==10.2.0.20240822 \
types-psutil==6.1.0.20241221 \
types-Pygments==2.19.0.20250107 \
types-python-dateutil==2.9.0.20241206 \
types-pytz==2025.1.0.20250204 \
types-PyYAML==6.0.12.20241230 \
types-requests==2.32.0.20241016 \
types-setuptools==75.8.0.20250210 \
types-tabulate==0.9.0.20241207 \
types-tqdm==4.66.0.20240417 \
types-ujson==5.10.0.20240515 > /dev/null 2>&1

#pip list
#which python
mypy .