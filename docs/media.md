# Interface recordings

The README includes screenshots, an upload walkthrough and an older Q&A recording. The pipeline diagrams and evaluation charts are separate from these demonstrations.

## Upload walkthrough

Captured on 12 October 2026 from the application at commit [609c0c8](https://github.com/urmeo/nexus-rag/commit/609c0c8472baffc65982af3c4f2021b7fa5c2f75), running locally in Chrome. The recording uses the real application and API without mocked responses or modified interface content.

| File | Content |
| --- | --- |
| [Desktop screenshot](../assets/figures/interface-desktop.png) | 1440 × 1000; one document indexed into six passages |
| [Mobile screenshot](../assets/figures/interface-mobile.png) | 390 × 844; the document drawer open |
| [Walkthrough](../assets/media/interface-walkthrough.mp4) | 21.1 seconds; upload, indexing, desktop view, mobile drawer and a question typed without submitting |

The input is the repository's [fictional sample paper](../examples/sample_paper.md): 301 words, indexed into six passages with `BAAI/bge-small-en-v1.5`. Data was stored in an isolated capture directory. Ollama was unavailable; the interface shows that state and no generated answer is presented.

Screenshots are direct browser captures. The video preserves the capture timing and fits each viewport inside a 1440 × 1000 canvas. White space around the mobile view keeps its proportions. It is an H.264 MP4 without audio.

## Historical Q&A

The [92-second Q&A recording](../assets/media/historical-qa.mp4) comes from `screenshots/nexusrag-demo.gif`, committed in [6ca6841](https://github.com/urmeo/nexus-rag/commit/6ca684149d6852b02b121de106d033d05b757f21). It shows a PDF upload followed by two questions and responses with citation markers. The interface displays MiniLM embeddings and `llama3.2:3b`; it predates the current BGE-small default and interface cleanup.

The source GIF is 720 × 335 and 95.3 seconds long. Conversion retains the first 92 seconds, omitting the final recording-toolbar frames, and pads the height to 336 pixels for H.264 encoding. The MP4 has no audio. The source GIF's SHA-256 is `37d41a4cc27ee854d76754c17509055ea50a82bda4a5d7dd3509ec7b4ed463a3`.

These responses are historical examples. Their correctness was not evaluated. Displayed confidence values and timings are not benchmark results. An [earlier static screenshot](../assets/figures/nexusrag-ui.png) is also retained; it shows an empty document list with the service disconnected.

## Playback

Both recordings are embedded as GitHub-hosted videos in the [README](../README.md#interface). Committed MP4 links provide a download option. The recordings do not replace the [evaluation protocol and recorded measurements](evaluation.md).
