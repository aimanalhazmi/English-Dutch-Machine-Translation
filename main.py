from src.preprocessing import preprocess_dataframe, get_tokenized_vocab
from src.load import load_data_as_df
from src.utils import split_dataset, get_embedding_models_paths
from src.embedding import PretrainedEmbeddingVocab

import torch
import wandb
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from src.dataset import TranslationDataModule
from src.model_lightning import Seq2SeqModel


def main():

    with wandb.init() as run:

        config = wandb.config
        run.name += "_" + "_".join([f"{key}_{config[key]}" for key in config.keys()])

        ## data pre-processing
        data_files = {
            "English": "data/europarl-v7.nl-en.en",
            "Dutch": "data/europarl-v7.nl-en.nl"
        }

        df = load_data_as_df(source_file=data_files[config.src_lang], source_col=config.src_lang, target_file=data_files[config.tgt_lang], target_col=config.tgt_lang)
        df_clean = preprocess_dataframe(df=df, source_col=config.src_lang, target_col=config.tgt_lang, config=config)
        df_sampled = df_clean.sample(frac=config.sample_frac, random_state=0).reset_index(drop=True)
        print(f"[Info] Selected {len(df_sampled):,} rows out of {len(df_clean):,} ({config.sample_frac * 100:.1f}% of preprocessed data)")
        train_df, val_df, test_df = split_dataset(df_sampled, val_test_size=config.val_test_size, test_size=config.test_size, random_state=0)

        src_vocab_set = get_tokenized_vocab(df=train_df, lan=config.src_lang)     # vocab should be built only from training data
        tgt_vocab_set = get_tokenized_vocab(df=train_df, lan=config.tgt_lang)

        src_emb_path, tgt_emb_path = get_embedding_models_paths(source_col=config.src_lang, target_col=config.tgt_lang, method=config.embedding_method)

        src_vocab = PretrainedEmbeddingVocab(embedding_path=src_emb_path, embedding_dim=config.embedding_dim, restrict_to_vocab=src_vocab_set)
        tgt_vocab = PretrainedEmbeddingVocab(embedding_path=tgt_emb_path, embedding_dim=config.embedding_dim, restrict_to_vocab=tgt_vocab_set)

        ## training
        # init data module
        data_module = TranslationDataModule(train_df, val_df, test_df, config.src_lang, config.tgt_lang, src_vocab, tgt_vocab, config.batch_size)

        # init model
        model = Seq2SeqModel(src_vocab,
                                tgt_vocab,
                                config.embedding_dim,
                                config.hidden_dim,
                                config.n_layers,
                                config.p_dropout,
                                src_vocab.vectors,
                                tgt_vocab.vectors,
                                config.trainable_embeddings,
                                config.teacher_forcing_ratio,
                                config.teacher_forcing_ratio_decay,
                                config.learning_rate)
        
        # init wandb logger
        logger = lightning.pytorch.loggers.WandbLogger(
            log_model="all"
        )

        # init trainer + callbacks
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)
        progress_bar_callback = TQDMProgressBar(refresh_rate=25)
        checkpoint_callback = ModelCheckpoint(
            dirpath = "checkpoints",
            filename = "seq2seq-loss-{val_loss:.2f}",
            save_top_k = 1,
            verbose = True,
            monitor = "val_loss",
            mode = "min"
        )

        # callback for debugging
        class DebugCallback(lightning.Callback):

            def __init__(self):
                super().__init__()

            def on_after_backward(self, trainer, model):
                # Check encoder embedding grads
                if model.encoder.embedding.weight.grad is not None:
                    grad = model.encoder.embedding.weight.grad
                    if model.frozen_src_embeddings is not None:
                        frozen_norm = grad[model.frozen_src_embeddings].norm().item()
                        trainable_norm = grad[[i for i in range(len(model.src_vocab)) if i not in set(model.frozen_src_embeddings.tolist())]].norm().item()
                        model.log("grad_norm_src_frozen", frozen_norm, on_epoch=True, on_step=False, prog_bar=False)
                        model.log("grad_norm_src_trainable", trainable_norm, on_epoch=True, on_step=False, prog_bar=False)

                # Check decoder embedding grads
                if model.decoder.embedding.weight.grad is not None:
                    grad = model.decoder.embedding.weight.grad
                    if model.frozen_tgt_embeddings is not None:
                        frozen_norm = grad[model.frozen_tgt_embeddings].norm().item()
                        trainable_norm = grad[[i for i in range(len(model.tgt_vocab)) if i not in set(model.frozen_tgt_embeddings.tolist())]].norm().item()
                        model.log("grad_norm_tgt_frozen", frozen_norm, on_epoch=True, on_step=False, prog_bar=False)
                        model.log("grad_norm_tgt_trainable", trainable_norm, on_epoch=True, on_step=False, prog_bar=False)

                # check lstm grads
                for name, param in model.encoder.rnn.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        model.log(f"grad_norm_encoder_lstm_{name}", grad_norm, on_epoch=True, prog_bar=False, on_step=False)

                for name, param in model.decoder.rnn.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        model.log(f"grad_norm_decoder_lstm_{name}", grad_norm, on_epoch=True, prog_bar=False, on_step=False)

        debug_callback = DebugCallback()

        trainer = lightning.Trainer(
            max_epochs = config.n_epochs,
            gradient_clip_val = config.gradient_clip_val,
            accelerator = "gpu" if torch.cuda.is_available() else "cpu",
            devices = 1,
            logger = logger,
            callbacks = [early_stopping_callback, progress_bar_callback, checkpoint_callback, debug_callback],
            log_every_n_steps = 10,
            check_val_every_n_epoch = 1
        )

        # train the model
        trainer.fit(model, data_module)


if __name__ == '__main__':

    main()



