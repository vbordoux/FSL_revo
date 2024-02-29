import pandas as pd
import csv
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from utils import init_seed, evaluate_prototypes_AVES, evaluate_prototypes, eval_to_Raven
from evaluation import evaluate
from glob import glob
import h5py
from post_proc_new import post_processing
from pathlib import Path
import datetime
from Model import AvesClassifier, ResNet

# Github token: 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

@hydra.main(config_name="config")
def main(conf : DictConfig):

    # Set some path and variable
    apply_post_processing = False
    prediction_path = os.path.join(conf.path.root_dir,f'Prediction/{conf.path.test_name}{conf.features.call_type}.csv')
    post_process_prediction_path = os.path.join(conf.path.root_dir,f'Prediction/post_processed_{conf.path.test_name}_{conf.features.call_type}.csv')

    # Load a pretrained model
    model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
    model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
    emb_dim = 768
    encoder = AvesClassifier(model_config_path, model_path, trainable=False, embedding_dim=emb_dim, sr=conf.features.sr)
    encoder.to(device)
    encoder.eval()

    if conf.set.features:
        # Generate samples on the fly for training and validation set

        pass

    if conf.set.train:
        print("========== TRAINING STARTING =========")
        device = 'cuda'
        init_seed()
    
        if conf.train.encoder == 'AVES':
            

            pass

    if conf.set.eval:
        print("========== EVALUATION STARTING =========")
        device = 'cuda'
        init_seed()

        if conf.train.encoder == 'Resnet':
            name_arr = np.array([])
            onset_arr = np.array([])
            offset_arr = np.array([])
            all_feat_files = [file for file in glob(os.path.join(conf.path.feat_test,'*.h5'))]

            for feat_file in all_feat_files:
                feat_name = feat_file.split('/')[-1]
                audio_name = feat_name.replace('h5','wav')

                print("Processing audio file : {}".format(audio_name))

                hdf_eval = h5py.File(feat_file,'r')
                strt_index_query =  hdf_eval['start_index_query'][:][0]

                with torch.no_grad():
                    onset,offset = evaluate_prototypes(conf,hdf_eval,device,strt_index_query)
                
                name = np.repeat(audio_name,len(onset))
                name_arr = np.append(name_arr,name)
                onset_arr = np.append(onset_arr,onset)
                offset_arr = np.append(offset_arr,offset)

        elif conf.train.encoder == 'AVES':
            name_arr = np.array([])
            onset_arr = np.array([])
            offset_arr = np.array([])

            wav_files_path = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set/GR'
            # annot_files_path = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set'

            # wav_files_path = '/home/reindert/Valentin_REVO/Data/Belle_project/wav_modif'
            annot_files_path = '/home/reindert/Valentin_REVO/FSL_revo/Ground truth/POS'

            # TODO change glob to listdir
            all_wav_files = [file for file in glob(os.path.join(wav_files_path,'*.wav'))]
            all_annot_files = [file for file in glob(os.path.join(annot_files_path,'*.txt'))]
            all_wav_files.sort()
            all_annot_files.sort()

            if not len(all_wav_files)==len(all_annot_files):
                print('ERROR: the number of wav files and annotation files does not match')
            
            min_length_list = []
            for wav_file, annot_file in zip(all_wav_files, all_annot_files):
                with torch.no_grad():
                    onset,offset, min_length = evaluate_prototypes_AVES(conf, encoder, device, wav_file, annot_file)

                name = np.repeat(os.path.basename(wav_file),len(onset))
                name_arr = np.append(name_arr,name)
                onset_arr = np.append(onset_arr,onset)
                offset_arr = np.append(offset_arr,offset)
                min_length_list.append(min_length)
        
            
        else:
            print('ERROR: other model than Resnet or AVES not supported now (change model in config.yaml)')

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        df_out.to_csv(prediction_path,index=False)
        print("File saved at ", prediction_path)



        if apply_post_processing:
            # Find the pos_event with the smallest duration to use as minimal in the post processing
            smallest_pos_sample_length = min(np.array(min_length_list))
            smallest_pos_sample_length

            post_processing(prediction_path, post_process_prediction_path, smallest_pos_sample_length)

            # Count annotation before and after postprocessing
            print(f"Smallest positive annotation: {smallest_pos_sample_length}s")
            df = pd.read_csv(prediction_path, delimiter=',', header=0)
            print("Count prediction:", len(df.index))
            new_df = pd.read_csv(post_process_prediction_path, delimiter=',', header=0)
            print("Count prediction after post-process: ", len(new_df.index))



    if conf.set.to_raven:
        # Function to transform a csv file (from prediction or annotation) into a txt file compatible with Raven

        # To create raven annotation for a specific file
        # post_process_prediction_path = '/home/reindert/Valentin_REVO/DCASE_2022/Development_Set/Validation_Set/PB/BUK5_20180921_015906a.csv'
        eval_to_Raven(conf, post_process_prediction_path)
    


    if conf.set.metric_eval:

        TEST_NAME = f'{conf.features.call_type}_{conf.path.test_name}'
        SAVE_RESULT = True # Save the result in evaluation_summary.csv
        gt_dir = conf.path.ground_truth_dir

        # Call the evaluation metric here.
        # Later I want to evaluate the AUC of the detector / binary classifier
        dest_path = './Eval_report/' 
        Path(dest_path).mkdir(exist_ok=True)

        # Find the prediction in the post process prediction file
        if apply_post_processing:
            pred_filepath = post_process_prediction_path
        else:
            pred_filepath = prediction_path
        
        print("\nEvaluation of the prediction in file: ", pred_filepath)
        # compute score from the metric
        overall_scores = evaluate(conf, pred_filepath, gt_dir, team_name='Test on', dataset='GR', savepath=dest_path, write_report=False)

        # Add the result as a new line in evaluation_summary.csv
        if SAVE_RESULT:
            # Load the result summary csv
            csv_file_path = '/home/reindert/Valentin_REVO/FSL_revo/evaluation_summary.csv'
            eval_hist_df = pd.read_csv(csv_file_path)

            # Add the new result into the summary csv
            result_df = pd.DataFrame(data={'Timestamp':'{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()),
                                'Test name':TEST_NAME,
                                'Call type':conf.features.call_type, 
                                'N shots':conf.train.n_shot, 
                                'Precision':overall_scores['precision'], 'Recall':overall_scores['recall'], 'Fmeasure':overall_scores['fmeasure (percentage)'],
                                'Threshold':conf.eval.threshold,
                                'Segment length':conf.features.seg_len,
                                'Post process': apply_post_processing},
                                index=[0])
            new_eval_hist_df = pd.concat([result_df, eval_hist_df], ignore_index=True)

            #Overwrite the existing csv with the new result added
            new_eval_hist_df.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    main()
