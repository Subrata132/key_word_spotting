class TrainingParams:
    feat_path = "./features_new"
    save_path = "./saved_results"
    model_name = "cnn_model_tiny_v1.0.h5"
    train_csv_loc = "./features/sa_train.csv"
    val_csv_loc = "./features/sa_val.csv"
    test_csv_loc = "./features/sa_test.csv"
    stat_csv_name = "stat_v1.2.csv"
    batch_size = 64
    lr = 1e-3
    max_epoch = 10
