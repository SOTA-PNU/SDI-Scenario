"""B522 v9: preserve v8, extend observed-track retention to the 3 s object horizon."""
import hashlib
from pathlib import Path
from b514_build_base import archive_before_write

ARENA=Path(__file__).resolve().parents[1]
SRC=ARENA/'base_carter_run_ros_map_lidartime_objects_check_v8.py'
OUT=ARENA/'base_carter_run_ros_map_lidartime_objects_check_v9.py'
SOURCE_SHA256='d92eed48917989427e226befd9db5c213e5ba2a3d3a8c1110304a0d525716fb6'


def build(source):
    assert hashlib.sha256(source.encode()).hexdigest()==SOURCE_SHA256,'v8 source changed'
    replacements=[
        ('vmax=2.0, keep=1.0, stale=0.3, frag_n=3, frag_join=0.6)',
         'vmax=2.0, keep=3.0, stale=0.3, frag_n=3, frag_join=0.6)'),
        ('1.0 s 넘게 보이지 않은 물체는 매 스텝 지웁니다.',
         '스캔 기준 시각(t - 0.1 s)에서 마지막 관측 후 3.0 s가 넘은 물체는 매 스텝 지웁니다.'),
        ('1.0 s 넘게 보이지 않은 물체는 빠집니다.',
         '스캔 기준 시각(t - 0.1 s)에서 마지막 관측 후 3.0 s가 넘은 물체는 빠집니다. 현재 age_s는 최대 약 3.1 s입니다.')]
    for before,after in replacements:
        assert source.count(before)==1,(before,source.count(before))
        source=source.replace(before,after,1)
    return source


if __name__=='__main__':
    source=build(SRC.read_text());compile(source,str(OUT),'exec')
    archive_before_write(OUT);OUT.write_text(source)
    print(OUT.name,hashlib.sha256(source.encode()).hexdigest())
