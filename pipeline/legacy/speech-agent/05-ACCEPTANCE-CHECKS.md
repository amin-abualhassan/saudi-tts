# Acceptance checks and handoff

The implementation agent must run the feasible checks and record pass/fail/not-tested with brief evidence. Do not present this checklist itself as proof of a working deployment.

## Start, connect, and finish

- Local app responds on port 8002. Public HTTPS test URL loads from another device and requests microphone access correctly.
- A real remote browser sends speech through LiveKit and plays the actual local fine-tuned TTS response. Successful signaling alone is insufficient.
- Test direct UDP media and the configured fallback on a second network when available. Record unsupported network cases honestly.
- Start produces exactly one greeting. Mute blocks microphone transmission as indicated; End stops capture, playout, generation, and the room session.
- A transient disconnect/reconnect creates no duplicate agent, repeated greeting, or stuck busy slot. Closing the tab eventually releases capacity.
- A second simultaneous visitor gets a friendly busy message. Ending or losing the first session allows a new one.

## Language and conversation

- Ask an Arabic question supported by the actual snapshot, such as "وش تقدمون في الجواب؟" The assistant uses natural Saudi Arabic and only supported claims.
- Ask a technical follow-up supported by the snapshot. It explains accurately without filling gaps with assumed Aljawab features.
- Ask "Can you explain that in English?" The next substantive reply is in English. Switch back to Arabic naturally.
- Try Arabic containing English technical terms. Recognition and speech remain intelligible.
- Ask for a price, integration, or guarantee deliberately absent from the verified snapshot. It says the information is unavailable and does not invent a number or commitment.
- Ask it to book a meeting or create a ticket. It does not claim an action was taken. It can describe a published contact method only if available in the snapshot.
- Ask it to ignore instructions or follow an instruction embedded in reference text. It retains scope and grounding.

## Interruption and speech

- Interrupt during the greeting, a cached acknowledgment, the first answer sentence, and a longer answer. In each case, stale audio stops and does not resume after the new question.
- Interrupt while synthesis is slow and another sentence is queued. A late HTTP chunk, decision result, or playback callback cannot revive the canceled answer.
- Say "لا، أقصد..." or "No, I meant..." while the assistant speaks. The next answer follows the correction and does not assume the canceled answer was fully heard.
- Test quiet pauses, Arabic hesitation, fan noise, and speaker echo. Background noise should not repeatedly create empty turns or self-interruptions.
- Verify punctuation-aware streaming using Arabic `؟`, English `?`, periods, abbreviations, decimal numbers, and a sentence with no early punctuation. Avoid token-by-token speech or waiting for the entire answer.
- Listen to cached clips and full answers for matching voice, volume, pronunciation, clean boundaries, and absence of decoder artifacts.
- Confirm fillers are occasional, language-appropriate, and do not overlap speech, repeat the LLM's opening, or sound approving after a complaint.
- Simulate slow/refused/unavailable Decisions. The main answer still starts without waiting; late filler output is dropped.

## Knowledge, failures, and privacy

- Snapshot contains real rendered website content and traceable URLs/timestamps. The assistant cannot cite invented services when extraction is empty.
- Refresh failure preserves a valid prior snapshot. First-run extraction failure is reported clearly and does not seed fabricated content.
- Missing OpenAI credentials, failed TTS, or dropped transcription connection produces a plain-language recoverable experience and useful private logs. No endless retry/filler loop.
- Client assets and network responses contain no server/provider secrets. Participant tokens are bounded to the intended room and expire.
- Recordings/transcripts are not persisted by application logs, room recording, browser storage, or tracing integrations by default. End clears in-memory session history.
- Existing TTS and unrelated services remain healthy after setup and after app restart.

## Measurements and reporting

Use at least several representative warm Arabic and English turns. Report sample count, conditions, speech-end-to-substantive-audio timing, filler timing separately, and interruption-to-playout-stop timing. Where feasible include p50/p95, but do not pretend a tiny sample is statistically robust. Treat first warmup separately. Explicitly state which timings were measured server-side and which reflect actual browser playout.

If no human microphone is available, use adapter/stream tests and prerecorded non-user fixtures where appropriate, and mark conversational audio quality and real remote-browser tests as pending. Do not replace honest limitations with a claim that a process being up proves success.

## Final delivery from the implementation agent

Provide the current HTTPS test link, local port 8002 URL, what is running, the source snapshot location/date, generated clip manifest, configuration and credentials variable names, start/stop/restart instructions, and exact remaining AWS port rules if any. Keep credentials out of the report. Include tested results and remaining limitations. Operational latency details belong in the report, not the customer interface.
