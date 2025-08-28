## Files Description
- `SEIR_model_general.py` is the code for SLIRD model
- `functions_general.py` contains the functions needed for `SEIR_model_general.py`
- `abcsmc_general.py` is the code for ABC-SMC calibration

## Running the Code
The main script for running the calibration process is not included in this repository. To run it, please create a script as in the example below:

```python
from abcsmc_general import run_calibration
region = 'London'
run_calibration(config_dir='./london_params.yaml',
                contact_matrix_dir=f'../data/regions/{region}/contacts_matrix/contacts_all_locations.csv',
                ifr_dir=f'../data/regions/{region}/epidemic/IFR_hetero.csv',
                filename='heteroContact_heteroIFR')
```
