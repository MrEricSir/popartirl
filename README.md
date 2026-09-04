# Pop Art IRL

[Try it out!](https://mrericsir.github.io/popartirl/)

HTML5 webapp that turns your webcam into a live pop art. Intended to run on phones, tablets, and laptops.

Uses edge detection (Sobel algorithm) with a reduced and blown-out color palette. Dots and stripes replace some of the colors to imitate comic panels and Roy Lichtenstein-style art.

## Features

- Full-screen live pop art/comic style art rendered on the fly from the camera.
- Long press to switch cameras. Front camera is mirrored by default.
- Adapts automatically to brightness/contrast changes.
- Double tap to show a QR code linking back to this page for sharing.
- Settings option on the QR code panel to tweak settings.

## Run

- [Run it in your browser](https://mrericsir.github.io/popartirl/)
- For local development, run `./dev.sh` and point your browser at
  `http://localhost:5500`, or pick another port with `PORT=<port> ./dev.sh`
  (camera access requires a webserver in most browsers.)

## About

- By [MrEricSir](https://mrericsir.github.io/)
- Pull requests and detailed bug reports are appreciated
- Available under an MIT license
