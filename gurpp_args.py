def get_default_arguments():
    return dict(
        data=dict(
            kg_dir='data/nymhtkg/mht180.csv',
            kg_reverse=False,
            kg_export=True,
            hg_feature_dir=['data/nymhtkg/kge_pretrained_transR/TransR_UrbanKG_1/UrbanKG_TransR_entity.npy',],
            kg_ent_pretrained_emb='data/nymhtkg/kge_pretrained_transR/TransR_UrbanKG_1/UrbanKG_TransR_entity.npy',
            image='data/nymhtkg/region_si_img.npy',
            flow=['data/nymhtkg/flow_in_per_hour.npy',
                  'data/nymhtkg/flow_out_per_hour.npy',
                  'aug',
                  ],
        ),

        gurp_model=dict(
            seed=2024,
            node_dict={'brand':0, 'cate1':1, 'junc_cate':2, 'junction':3, 'poi':4, 'region':5, 'road':6, 'road_cate':7},
            edge_dict={'JCateOf':0, 'BrandOf':1, 'Cate1Of':2, 'HasJunc':3, 'HasPoi':4, 'HasRoad':5, 'NearBy':6,
                       'RCateOf':7},
            node_dim=[144, 144, 144],
            n_layers=2,
            n_heads=4,
            batch_size=180,
            agg_method='add',
            use_norm=True,
            flow_dim=[144, 144, 144],
            out_dim=144
        ),

        gurp_training=dict(
            seed=1,
            log_folder='experiments/gurp_model',
            logger_name='train_gurp',
            log_level='INFO',
            log_attr='final',

            lr=0.001,
            weight_decay=1e-6,

            noise_std=0.05,
            sample_size=None,
            margin=2.0,

            mobility='data/nymhtkg/mobility_distribution.npy',
            ratio_dict={'sp_pos': 1, 'sp_neg': 1},
            aug_sel_ratio=0.15,
            batch_size=180,
            epochs=500,
            pred_loss_weight=0.01,

            save_model_path='experiments/gurp_model',
        ),

        gurp_prompt_training=dict(

            seed=2024,
            log_folder='experiments/gurp_prompt',
            logger_name='train_gurp_prompt_with_emb',
            log_level='INFO',

            pre_train_region_emb='experiments/gurp_model/XXXXXX', # path to pre-trained region embedding
            check_counts='data/task/check_counts.npy',
            crime_counts='data/task/crime_counts.npy',
            hop_k=2,
            sub_max_nsize=50,
            crime_epoch=200,
            check_epoch=200,
            model_save_dir='experiments/gurp_prompt',
        ),
    )