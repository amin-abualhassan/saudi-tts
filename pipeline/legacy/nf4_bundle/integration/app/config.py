from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    openai_api_key: str = ''
    stt_model: Literal['gpt-live-transcribe'] = 'gpt-live-transcribe'
    llm_model: Literal['gpt-6-luna'] = 'gpt-6-luna'
    tts_model: Literal['attention_mlp_r16_early'] = 'attention_mlp_r16_early'
    tts_checkpoint_sha256: Literal['9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf', 'bd78f331a5a1534cb8bfecc6c0a024db7c0c5267a6c9da806a2dde1e18dd29af'] = '9dbf9c8b6ee71dc5c737e6a76f2166539e833a6aa8ab274fb192553714a294cf'
    tts_url: str = 'http://127.0.0.1:3000'
    tts_reference: str = ''
    tts_reference_text: str = ''
    tts_reference_sha256: str = ''
    tts_voice_name: str = 'Noura نورة'
    app_secure_cookies: bool = False
    tts_studio_directory: Path = ROOT/'tts-data'
    livekit_api_key: str = ''
    livekit_api_secret: str = ''
    livekit_url: str = 'ws://127.0.0.1:7880'
    decisions_enabled: bool = True
    acknowledgment_enabled: bool = True
    decisions_deadline_ms: int = 800
    acknowledgment_probability_threshold: float = .8
    acknowledgment_cooldown_ms: int = 6000
    acknowledgment_eligibility_delay_ms: int = 350
    end_of_turn_silence_ms: int = 550
    vad_activation_threshold: float = .7
    vad_min_speech_duration_ms: int = 220

settings = Settings()
