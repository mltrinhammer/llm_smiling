# Using Large Language Models to Contextualize Smiling in Psychotherapy Research

Code for the manuscript *Using Large Language Models to Contextualize Smiling in Psychotherapy Research* (Trinhammer, Grasshof, Volkert, Rasche, Weiland, Brandt, & Altmann), currently being prepared for submission.

The pipeline annotates the sentiment of each participant speech turn in an attachment interview with an open-weight LLM, and analyzes individual smiling, simultaneous smiling, and cross-lagged smile synchrony (OpenFace AU06 + AU12) as a function of that sentiment. Everything runs locally on a single consumer GPU.

## Contents

`preprocessing/`
- `transcribe.py` — speaker diarization (pyannote 3.1) and transcription (Whisper), producing speech turns
- `translate.py` — German → English turn translation (Helsinki-NLP `opus-mt-de-en`)
- `sentiment_annotation.py` — per-turn sentiment labels (positive / neutral / negative / other) from Qwen2.5-14B via llama.cpp, with few-shot prompting and preceding-turn context

`analyses/`
- `common.py` — smile signal construction and per-turn measures shared by all analyses
- `analysis1.py` — individual smiling (frequency and intensity) by sentiment; LMM with within-participant permutation LRT
- `analysis2.py` / `analysis2_stats.py` — per-turn dyadic synchrony measures, then GLMM of simultaneous smiling by sentiment (Analysis 2a) and LMM of cross-lagged smile-intensity synchrony by sentiment (Analysis 2b)


## Data

The interview recordings, transcripts, and questionnaire data are not included and cannot be shared: they are pseudonymized clinical audio/video from human participants who consented to scientific analysis only. The code is published for transparency and reuse on comparable data.

## Contact

Anyone interested in this work is more than welcome to get in touch — about the method, the code, reuse on their own data, or possible collaboration. Write to the corresponding author, Martin Lund Trinhammer (<mlut@itu.dk>), or to any of the co-authors listed in the manuscript.
