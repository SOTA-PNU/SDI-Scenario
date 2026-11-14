# 이식 패키지에서 원본 커밋에 가한 변경

이 디렉터리는 `carter-scenario-levels` 커밋 `e184e99`의 사본이며, 아래 두 가지만 다르다.
둘 다 **원본 서버에서는 동작이 완전히 동일**하고(그 서버의 `$HOME`이 `/home/jun`이므로),
다른 서버에서만 의미가 생긴다.

## 1. `CARTER_WS` 기본값을 `$HOME/carter_ws`로

| 파일 | 원본 | 변경 |
|---|---|---|
| `levels/run_isaac.sh:9` | `${CARTER_WS:-/home/jun/carter_ws}` | `${CARTER_WS:-$HOME/carter_ws}` |
| `levels/common/carter_env.py:37` | `os.environ.get("CARTER_WS", "/home/jun/carter_ws")` | `os.environ.get("CARTER_WS", os.path.expanduser("~/carter_ws"))` |
| `levels/common/probe_scene.py:253` | 위와 같음 | 위와 같음 |

이유: 새 서버에는 `/home/jun`이 없다. 기본값이 그대로면 `CARTER_WS`를 export하지 않고
`run_isaac.sh`를 직접 부를 때 "브리지 rclpy를 못 찾음" 같은 엉뚱한 에러로 나타난다.
`carter-run`은 항상 `CARTER_WS`를 export하므로 래퍼를 쓰면 어느 쪽이든 차이가 없다.

## 2. `REMOTE_VIEWING.md`의 호스트 주소 치환

원본 서버의 사설/공인 IP를 `<SOURCE_HOST_LAN_IP>` / `<SOURCE_HOST_PUBLIC_IP>`로 바꿨다.
새 서버에서는 어차피 틀린 값이고, 내부망 주소를 그대로 실어 보낼 이유가 없다.
원격 뷰잉은 벤치마크 재현에 필요 없다 (헤드리스로 돌아간다).

## 그 밖에 제외한 것

데모 영상(`*.mp4`)과 발표자료(약 190 MB)는 용량만 차지하고 재현과 무관하므로 뺐다.
필요하면 GitHub 원본 리포에서 받으면 된다.

**소스 코드와 기준 결과 JSON은 한 글자도 손대지 않았다.**
