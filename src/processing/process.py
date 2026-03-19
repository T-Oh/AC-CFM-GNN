import os

from utils.processing import ProcessingConfig
from processing.process_ac import process_ac
from processing.process_zhu_n_minus_k import process_zhu_n_minus_k
from processing.process_ldtsf import process_ldtsf
from processing.process_ldtsf_dc import process_ldtsf_dc
from processing.process_n_minus_k import process_n_minus_k

def run_processing(cfg):

    if not os.path.exists(os.path.join(cfg['dataset::path'], 'processed')):
        os.mkdir(os.path.join(cfg['dataset::path'], 'processed'))
    data_type = cfg['data']
    root = cfg['dataset::path']
    raw_paths = [os.path.join(cfg['dataset::path'], 'raw', f) for f in os.listdir(os.path.join(cfg['dataset::path'], 'raw'))]
    processed_dir = os.path.join(cfg['dataset::path'], 'processed')
    PROCESSING_CONFIG = ProcessingConfig(root=root, raw_paths=raw_paths, processed_dir=processed_dir, data_type=cfg['data'], edge_attr_type=cfg['edge_attr'], ls_threshold=cfg['ls_threshold'], N_below_threshold=cfg['N_below_threshold'],
                                                normalize_injection=cfg['normalize_injection'], multiply_base_voltage=cfg['multiply_base_voltage'], zhu_check_buses=cfg['zhu_check_buses'], check_s_y=cfg['check_s_y'])

    if cfg['process']:
        print('Processing...')
        if data_type in ['AC', 'LSTM', 'Zhu', 'Zhu_mat73', 'ANGF_Vcf', 'Zhu_nobustype']:
            if data_type == 'LSTM':    PROCESSING_LSTM_DATA = True
            process_ac(PROCESSING_CONFIG)
        elif data_type == 'Zhu_n_minus_k':
            process_zhu_n_minus_k(PROCESSING_CONFIG)
        elif data_type == 'LDTSF':
            process_ldtsf(PROCESSING_CONFIG)
        elif data_type == 'LDTSF_DC':
            process_ldtsf_dc(PROCESSING_CONFIG)
        elif data_type == 'n-k':
            process_n_minus_k(PROCESSING_CONFIG)
        else:
            assert False, 'Datatype must be one of the following: AC, LSTM, Zhu, Zhu_mat73, ANGF_Vcf, Zhu_nobustype, Zhu_n_minus_k, LDTSF, LDTSF_DC, n-k!'
        print('Processing finished!')


