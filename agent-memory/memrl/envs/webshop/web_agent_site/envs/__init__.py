from gym.envs.registration import register

try:
    from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
except ImportError:
    WebAgentSiteEnv = None

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

if WebAgentSiteEnv is not None:
    register(
      id='WebAgentSiteEnv-v0',
      entry_point='web_agent_site.envs:WebAgentSiteEnv',
    )

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)