# PRO 6000 서버 이전 (2026-10-28)

fractal 이 자주 꺼져서 프로젝트 C 의 B 실험을 PRO 6000 서버(giga-w773-s-3, <server-ip>)로 옮긴다.
두 서버 모두 사용자 `jun`, 홈 `/home/jun` 이라 경로는 그대로 둔다.

| 무엇 | 어떻게 | 크기 |
|---|---|---|
| `~/SDI-Scenario` (영상과 프레임 이미지 제외) | fractal 에서 rsync | 약 1.5 GB |
| `~/carter_ws` (Isaac Sim 4.5.0.0 micromamba 환경, 자체 Python 3.10, carter-run) | fractal 에서 rsync | 16 GB |
| `~/furio-gpu/venv` (vllm 0.32.0, torch 2.13.0, transformers 5.19.0) | 그 서버에서 새로 설치: `setup_vllm_and_model.sh` | 약 17 GB |
| `~/models/gpt-oss-120b` (리비전 b5c939de, 파일 22개) | 그 서버가 Hugging Face 에서 받고 해시 대조 | 61 GB |
| `~/models/serve_gpt-oss-120b.sh` | 이 폴더에서 복사 | - |

- fractal 의 vLLM venv 는 시스템 Python 3.12 (Ubuntu 24.04) 위에 있어 22.04 로는 복사가 안 된다. 그래서
  `vllm_venv_lock.txt` (fractal venv 의 206개 패키지 그대로) 로 uv 가 관리하는 CPython 3.12.3 위에 다시 만든다.
- `gpt-oss-120b_SHA256SUMS` 는 fractal 의 파일 해시다 (가중치 15개는 Hugging Face LFS etag).
- 접근: fractal 에서만 그 서버로 닿는다 (반대는 안 됨). fractal 전용 키 `~/.ssh/id_ed25519_pro6000` 를 그 서버의
  `~/.ssh/authorized_keys` 에 등록했다 (사용자 승인). 이전이 끝나면 지울 수 있다.
- rsync 는 실험실 망을 막지 않도록 속도를 제한한다.
