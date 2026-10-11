"""Build the downloadable source archive, excluding credentials/runtime caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo
root=Path(__file__).resolve().parents[1]
allowed={'.github','dist','docs','ml','scripts','tests'}
with ZipFile(root/'dist/source.zip','w',ZIP_DEFLATED) as out:
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if '__pycache__' in rel.parts or p.suffix=='.pyc' or rel.as_posix()=='dist/source.zip':continue
        if len(rel.parts)>1 and rel.parts[0] not in allowed and rel.as_posix()!='.openai/hosting.json':continue
        if p.name.startswith('.env') or p.suffix in ['.tar','.gz']:continue
        z=ZipInfo('baywatch/'+rel.as_posix(),date_time=(2026,10,11,0,0,0));z.compress_type=ZIP_DEFLATED
        out.writestr(z,p.read_bytes())
print('Created dist/source.zip')
