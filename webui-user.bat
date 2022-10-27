@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--deepdanbooru --xformers --force-enable-xformers --opt-split-attention --gradio-img2img-tool color-sketch --autolaunch --theme dark --disable-safe-unpickle --opt-channelslast --allow-code
call webui.bat
