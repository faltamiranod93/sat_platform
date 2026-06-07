import os
from pathlib import Path

# Override con variable de entorno para el computador de la universidad:
#   Windows:  setx MSC_UTFSM_ROOT "D:\investigacion\Msc-UTFSM"
#   Linux:    export MSC_UTFSM_ROOT=/home/user/Msc-UTFSM
MSC_ROOT = Path(os.environ.get('MSC_UTFSM_ROOT', r'C:/Users/felip/Desktop/Msc-UTFSM'))

# Estos archivos viven en el repo — path relativo a este archivo
CONFIG_JSON = Path(__file__).parent / 'config' / 'config_bandas_v3.json'
MCAL_DIR    = Path(__file__).parent / 'data' / 'Laguna-Seca'
