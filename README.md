# NOVA — Offline Vision Assistant for People Who Are Blind or Have Low Vision

NOVA turns a camera (or a recorded video) into spoken, live awareness of the
surroundings: what is there, where it is, what colour it is, whether it is getting
closer, and where an object was last seen. Everything runs on the user's own
computer — there are no cloud services at runtime.

## What it does

- **Detects and tracks objects** live (YOLOv8 with tracking; detections below 0.35 confidence are ignored).
- **Says where things are**: left / centre / right, plus *approaching* or *moving away*.
- **Proximity labels** (e.g. Close / Far) and spoken **danger warnings** for things coming toward the user.
- **Colour detection** for each object (K-means on the object's pixels).
- **Memory**: remembers where useful objects were last seen and answers
  *"Where is my phone?"* later.
- **Read Text (OCR)**: reads text from the camera view (Arabic + English) and pulls out
  amounts, phone numbers, dates, emails and links.
- **Ask AI**: a local Llama model (via Ollama) answers general questions. Questions about
  the scene and memory are routed to NOVA's own tools first.
- **Spoken output** through the system text-to-speech voice.
- Optional **depth-map overlay** (MiDaS) for visualisation.

## How it works

```
camera / video -> YOLOv8 tracking -> direction, approach status, proximity
                                  -> colour (nova_color_organizer.py)
                                  -> memory (nova_memory.py)
question -> agent (nova_agent_llm.py): NOVA tools first, local Llama for the rest
image    -> OCR (nova_ocr.py, EasyOCR)
UI       -> nova_ui.py   (Tkinter)
```

## Setup

1. Python 3.11 and `pip install -r requirements.txt`
2. Install [Ollama](https://ollama.com) and run `ollama pull llama3.2:3b`
3. First run downloads model weights once (YOLOv8s, EasyOCR language models, MiDaS if depth is used).
   After that no internet connection is needed.

## Run

```
python nova_ui.py
```

The app can use the webcam or a recorded video file.

## Test results

Measured on one laptop: Intel Core i3-1115G4 @ 3.00 GHz, 8 GB RAM, Intel UHD Graphics
(no discrete GPU), Windows 10, CPU only. Times were taken by hand with a stopwatch
(about +/- 0.5 s); FPS is the counter shown in the app.

| Measure | Result |
|---|---|
| Memory question ("Where is my phone?") in the Memory tab | about 0.5 s |
| Same memory question asked in the AI tab | about 0.5 s |
| General question answered by the local Llama model (AI tab) | about 30 s |
| Read Text (OCR) on one camera frame | about 22 s |
| Frames per second (CPU only, on-screen counter) | about 4 FPS |
| Memory recall: correct answers / attempts | 20 correct out of 20 |
| Objects detected correctly in the demo video | 2 correct out of 2 checked (very small sample) |

Notes: the Llama time may include loading the model into memory on the first question, and
it grows with the length of the answer. The OCR time may include loading the OCR models
on the first read.

## Known limitations

- **Object classes come from the COCO set (80 classes).** It has no class for keys or a
  wallet, so NOVA cannot find those. Phones, bottles, cups, laptops, bags and books work.
- **"Approaching" and Close/Far are estimated from how the bounding box grows.** They are not
  metric distances. The depth map is relative and only a visual aid.
- **The agent is not a fully autonomous LLM agent.** Keyword rules route scene and memory
  questions to NOVA's tools; the local Llama model handles everything else and cannot see
  the camera.
- Colour names are approximate and degrade in poor light; blurry or dark video lowers detection accuracy.
- Live-webcam support depends on the camera driver; most testing used recorded video.
- On a CPU-only laptop the frame rate is limited, general questions to the local Llama model
  are slow (see the table above), and the first OCR read takes tens of seconds.

## AI assistance and credits

This project was built with the help of AI assistants (Claude, ChatGPT and Gemini) for coding,
code review and planning. The team ran, tested and is responsible for the code.

- Project lead, detection pipeline, memory, agent, UI: Yassin Adnan Mohemad Abdel Mohsen
- OCR module: Yassin Mohemad Ahmed Amin
