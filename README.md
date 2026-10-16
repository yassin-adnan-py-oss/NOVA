# NOVA — Offline Vision Assistant for People Who Are Blind or Have Low Vision

NOVA turns a camera (or a recorded video) into spoken, real-time awareness of the
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

*Fill this table with real measurements only (machine: ____________).*

| Measure | Result |
|---|---|
| Frames per second (CPU only) | |
| Time to answer a memory question | |
| Memory recall: correct answers / attempts | |
| Objects detected correctly in the demo video | |

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
- On a CPU-only laptop the frame rate is limited.

## AI assistance and credits

This project was built with the help of AI assistants (Claude, ChatGPT and Gemini) for coding,
code review and planning. The team ran, tested and is responsible for the code.

- Project lead, detection pipeline, memory, agent, UI: [YOUR NAME]
- OCR module: [TEAMMATE NAME]

