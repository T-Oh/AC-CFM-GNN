import os

from utils.processing import ProcessingConfig
from processing.process_ac import process_ac
from processing.process_zhu_n_minus_k import process_zhu_n_minus_k
from processing.process_ldtsf import process_ldtsf
from processing.process_ldtsf_dc import process_ldtsf_dc
from processing.process_n_minus_k import process_n_minus_k

def run_processing(cfg_processing: ProcessingConfig):

    if not os.path.exists(os.path.join(cfg_processing.root, 'processed')):
        os.mkdir(os.path.join(cfg_processing.root, 'processed'))


    print('Processing...')
    if cfg_processing.data_type in ['AC', 'LSTM', 'Zhu', 'Zhu_mat73', 'ANGF_Vcf', 'Zhu_nobustype']:
        if cfg_processing.data_type == 'LSTM':    PROCESSING_LSTM_DATA = True
        process_ac(cfg_processing)

    elif cfg_processing.data_type == 'Zhu_n_minus_k':
        process_zhu_n_minus_k(cfg_processing)
    elif cfg_processing.data_type == 'LDTSF':
        process_ldtsf(cfg_processing)
    elif cfg_processing.data_type == 'LDTSF_DC':
        process_ldtsf_dc(cfg_processing)
    elif cfg_processing.data_type == 'n-k':
        process_n_minus_k(cfg_processing)
    else:
        assert False, 'Datatype must be one of the following: AC, LSTM, Zhu, Zhu_mat73, ANGF_Vcf, Zhu_nobustype, Zhu_n_minus_k, LDTSF, LDTSF_DC, n-k!'
    print('Processing finished!')



