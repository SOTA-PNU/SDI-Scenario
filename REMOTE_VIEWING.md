# REMOTE_VIEWING — 내 컴퓨터에서 시뮬레이터 화면 보기

이 서버(A100)는 headless라 모니터가 없다. SSH로 쓰는 원격 사용자가 화면을 보는 방법은 두 가지다.

| 방법 | 조건 | 결론 |
|---|---|---|
| **A. 녹화 → MP4 받기** | 없음 (항상 됨) | **권장** — 아래 §1 |
| **B. WebRTC 라이브 스트리밍** | 노트북에서 서버로 TCP 49100 **+ UDP 47998** 직접 도달 | 내부망/포트포워딩 전용 — §2 |

## 1. 녹화해서 보기 (권장)

하네스(`levels/common/carter_env.py`)에 opt-in 녹화 훅이 있다. 환경변수만 켜면
런 중 뷰포트가 PNG로 저장된다 (러너 코드 수정 불필요):

```bash
cd /tmp/cv-infra-carter-levels
CARTER_RECORD=1 bash levels/run_isaac.sh levels/level2/solution_carter_run.py
# -> levels/level2/results/frames_solution/00000.png ...
```

환경변수 옵션:

| 변수 | 기본값 | 의미 |
|---|---|---|
| `CARTER_RECORD` | (off) | 아무 값이나 설정하면 녹화 on |
| `CARTER_RECORD_DIR` | `results/frames_<variant>` | 프레임 저장 경로 재지정 |
| `CARTER_RECORD_EVERY` | `2` | N 프레임마다 1장 캡처 (물리 60 fps → 30 fps 영상) |
| `CARTER_RECORD_EYE` | `-1.0,-2.5,7.5` | 카메라 위치 (x,y,z) — 기본값은 주행선 전체가 보이는 부감 |
| `CARTER_RECORD_TARGET` | `-6.0,2.0,0.3` | 카메라가 바라보는 점 |
| `CARTER_RECORD_SIZE` | `1280x720` | 렌더 해상도. **고화질은 `1920x1080` 권장** — 아래 참고 |

**고화질 레시피**: A100은 RT 코어가 없고 DLSS-RR 디노이저도 미지원이라 렌더에 스펙클
노이즈가 낀다. `CARTER_RECORD_SIZE=1920x1080` 으로 크게 렌더한 뒤 ffmpeg에서 1280으로
**다운스케일(슈퍼샘플링)** 하면 노이즈가 평균화돼 원본 자체가 깨끗해지고 압축 효율도 좋아진다:

```bash
CARTER_RECORD=1 CARTER_RECORD_SIZE=1920x1080 \
  bash levels/run_isaac.sh levels/level2/solution_carter_run.py
cd levels/level2/results
~/.local/bin/ffmpeg -framerate 30 -i frames_solution/%05d.png \
  -vf "tmix=frames=5,hqdn3d=4:3:8:6,scale=1280:-2" -c:v libx264 -pix_fmt yuv420p \
  -crf 19 -preset slow l2_solution_hq.mp4
```

`tmix=frames=5`(5프레임 시간축 평균)가 핵심이다: 카메라가 고정이라 정지 영역의 스펙클이
프레임 평균으로 거의 사라진다 (움직이는 로봇에는 자연스러운 모션블러가 생기는 정도).
A/B 실측: atadenoise 단독보다 tmix+hqdn3d 조합이 확연히 깨끗했다.

MP4 인코딩 (ffmpeg 정적 바이너리가 `~/.local/bin/ffmpeg`에 있음):

```bash
cd levels/level2/results
~/.local/bin/ffmpeg -framerate 30 -i frames_solution/%05d.png \
  -vf "hqdn3d=8:6:12:9,scale=854:-2" -c:v libx264 -pix_fmt yuv420p \
  -crf 28 -preset slow l2_solution.mp4
```

> denoise(`hqdn3d`)가 중요하다: A100은 RT 코어가 없어 렌더에 스펙클 노이즈가 끼고,
> 이게 x264 압축률을 크게 망친다 (같은 화질 기준 5배 이상 커짐).

내 컴퓨터로 가져오기 (노트북 터미널에서):

```bash
scp -P 10022 jun@164.125.19.138:/tmp/cv-infra-carter-levels/levels/level2/results/l2_solution.mp4 .
```

⚠ **녹화 런은 시각화 전용이다.** 캡처 오버헤드로 타이밍이 1프레임쯤 밀린다
(실측 예: L2 solution ttg 15.250 → 15.233 s). 문서화된 벤치마크 결과
(`results/*_result.json`/CSV)를 녹화 런이 덮어썼다면 `git checkout -- <파일>`로 복원할 것.
프레임 폴더(`frames_*/`, GB 단위)와 mp4는 `.gitignore`로 커밋에서 제외되어 있다.

## 2. WebRTC 라이브 스트리밍 (조건부)

하네스에 opt-in 훅이 있다:

```bash
CARTER_LIVESTREAM=1 CARTER_LIVE_WAIT=60 bash levels/run_isaac.sh levels/level2/solution_carter_run.py
```

`CARTER_LIVE_WAIT=60`: 부팅 완료 후 미션 시작 전에 60초(벽시계) 동안 로봇이 대기한다 —
이 사이에 클라이언트로 접속하면 주행 시작부터 볼 수 있다 (판정에는 영향 없음).

부팅 로그에 `[env] WebRTC livestream enabled` 가 뜨고 서버가 **TCP 49100**(시그널링)으로
리스닝한다. 시청은 NVIDIA **"Isaac Sim WebRTC Streaming Client"** (Windows/Linux/macOS,
NVIDIA 사이트에서 배포)에 서버 IP를 넣고 접속.

**네트워크 제약 — 이게 핵심이다:**

- 필요 포트: **TCP 49100** (시그널링) + **UDP 47998** (미디어).
- 이 서버의 실제 IP는 사설망 `10.254.182.72` 이고, 외부에서는 `164.125.19.138:10022`
  SSH 포워딩으로만 들어온다. 즉 **노트북이 같은 내부망(10.254.x.x 대역)에 있거나**,
  관리자가 두 포트를 외부로 포워딩해 줘야 라이브 시청이 된다.
- SSH 터널(`ssh -L 49100:...`)로는 TCP 시그널링만 넘어가고 **UDP 미디어는 터널을 못
  넘는다** → 접속은 되는데 화면이 안 나오는 형태로 실패한다.

같은 내부망에서의 접속 예: 클라이언트에 `10.254.182.72` 입력 → 연결.
(방화벽 확인: 서버에서 `ss -tln | grep 49100` 으로 리스너 확인 가능.)

## 3. 요약

- 외부(SSH만 가능)에서는 **§1 녹화**가 사실상 유일한 방법이고, 품질도 충분하다
  (1280×720 캡처, 30 fps 실시간 재생).
- 내부망에 앉을 수 있으면 §2 라이브가 된다 — 하네스는 준비돼 있다.
