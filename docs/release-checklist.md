# Release Checklist

Manual verification steps for each tagged release:

* [ ] `docker pull ghcr.io/lan-transcriber:$TAG`
* [ ] `docker run -p 7860:7860 -v $PWD/voices:/root/voices ghcr.io/lan-transcriber:$TAG`
* [ ] Upload a WAV file longer than two minutes recorded at 48 kHz.
* [ ] Check that the markdown transcript and summary are displayed.
* [ ] Rename a speaker alias and confirm the UI updates immediately.
* [ ] Download `summary.md` and open it on your machine.

Sign-off by: _______________

