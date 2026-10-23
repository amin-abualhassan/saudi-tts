from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    openai_api_key: str = ''
    stt_model: Literal['gpt-live-transcribe'] = 'gpt-live-transcribe'
    llm_model: Literal['gpt-6-luna'] = 'gpt-6-luna'
    tts_model: Literal['attention_mlp_r16_early'] = 'attention_mlp_r16_early'
    tts_url: Literal['http://127.0.0.1:3000'] = 'http://127.0.0.1:3000'
    tts_reference: str = ''
    tts_reference_text: str = ''
    tts_reference_sha256: str = ''
    tts_voice_name: str = 'Noura نورة'
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
    session_idle_timeout_seconds: float = Field(default=60,gt=0)
    session_start_limit_per_minute: int = Field(default=60,ge=1,le=1000)
    audio_dynamics_enabled: bool = True
    audio_peak_dbfs: float = Field(default=-6,ge=-24,le=-1)
    audio_quiet_boost_db: float = Field(default=3,ge=0,le=6)
    audio_onset_ms: float = Field(default=40,ge=0,le=120)
    audio_onset_attenuation_db: float = Field(default=-12,ge=-24,le=0)
    audio_leveler_enabled: bool = True
    audio_leveler_window_ms: float = Field(default=250,ge=100,le=1000)
    audio_leveler_target_dbfs: float = Field(default=-24,ge=-36,le=-12)
    audio_leveler_max_boost_db: float = Field(default=9,ge=0,le=12)

settings = Settings()
