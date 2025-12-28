```python

# Import and setup scene
blender --background --python import_tank_blender.py -- \
    --asset-dir /path/to/tank \
    --output /path/to/tank_scene.blend \
    --setup-scene

# Then render multi-view with the previous script
blender --background /path/to/tank_scene.blend \
    --python blender_multiview.py -- \
    --output ./renders/tank \
    --resolution 1024


```



```bash

# Convert DDS to PNG
cd /home/lockin/Projects/vehicle_insertion/src/assets/3d_models/t-90/source
python3 -c "
from PIL import Image
import glob

for dds in glob.glob('*.dds'):
    print(f'Converting {dds}...')
    img = Image.open(dds)
    img.save(dds.replace('.dds', '.png'), 'PNG')
    print(f'  -> {dds.replace(\".dds\", \".png\")}')
"

```