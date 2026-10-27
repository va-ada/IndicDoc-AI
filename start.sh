#!/usr/bin/env bash
# IndicDoc AI - start everything on a laptop with one command:
#   bash start.sh
# Checks Docker (installs Docker Desktop with Homebrew on macOS if missing), downloads the layout model and the
# OCR language models once, builds the image, starts the web app (:8501) and the API (:8000) and opens the browser.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- Docker
if ! command -v docker >/dev/null 2>&1; then
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    say "Installing Docker Desktop (Homebrew)"
    brew install --cask docker
  else
    fail "Docker is not installed. Install Docker Desktop from https://www.docker.com/products/docker-desktop/ and run this again."
  fi
fi

if ! docker info >/dev/null 2>&1; then
  say "Starting Docker"
  if [[ "$(uname)" == "Darwin" ]]; then
    open -a Docker
  elif command -v systemctl >/dev/null 2>&1; then
    sudo systemctl start docker || true
  else
    echo "Start Docker Desktop now (Start menu); waiting for it ..."
  fi
  for _ in $(seq 1 90); do docker info >/dev/null 2>&1 && break; sleep 2; done
  docker info >/dev/null 2>&1 || fail "Docker did not start. Open Docker Desktop, wait until it is running, then run: bash start.sh"
fi

if docker compose version >/dev/null 2>&1; then COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then COMPOSE=(docker-compose)
else fail "docker compose is missing; update Docker Desktop."; fi

# ---------------------------------------------------------------- models (once)
fetch() {  # url dest
  [[ -s "$2" ]] && return 0
  echo "  downloading $(basename "$2") ..."
  mkdir -p "$(dirname "$2")"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}
say "Models (downloaded once, about 300 MB)"
fetch "https://zenodo.org/records/15881917/files/doclayout_yolo_indicdlp.pt?download=1" \
      outputs/models/indicdlp_authors/doclayout_yolo_indicdlp.pt
for l in asm ben eng guj hin kan mal mar ori pan tam tel; do
  fetch "https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata" "outputs/models/tessdata/$l.traineddata"
done

# ---------------------------------------------------------------- build and run
say "Building and starting IndicDoc AI (the first build takes 5-10 minutes)"
"${COMPOSE[@]}" up --build -d

say "Waiting for the web app"
for _ in $(seq 1 90); do curl -fs http://localhost:8501/_stcore/health >/dev/null 2>&1 && break; sleep 2; done
curl -fs http://localhost:8501/_stcore/health >/dev/null 2>&1 || fail "The app did not come up. See: ${COMPOSE[*]} logs app"

URL=http://localhost:8501
case "$(uname)" in
  Darwin) open "$URL" ;;
  Linux) xdg-open "$URL" >/dev/null 2>&1 || true ;;
  *) start "$URL" 2>/dev/null || true ;;
esac
cat <<EOF

IndicDoc AI is running.
  Web app : $URL
  REST API: http://localhost:8000/docs

Try it: Verify Paper -> "Translation against its source"
  paper : docs/test_papers/02_hindi_number_changed/hindi_version.pdf
  source: docs/test_papers/02_hindi_number_changed/english_source.pdf
  expect: Q15 flagged (45 -> 102). All tests: docs/test_papers/ANSWER_KEY.md

Stop: ${COMPOSE[*]} down
EOF
