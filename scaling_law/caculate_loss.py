import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
import os
from llmshearing.datasets.streaming_dataset import TextStreamingDataset
from torch.cuda.amp import autocast
from llmshearing.data.llama_tokenizer import Tokenizer
from llmshearing.utils.test_composer_hf_eq import construct_example_cfg, ComposerMosaicLlama
from omegaconf import OmegaConf as om

class HFModel:
    def __init__(self, model_path, tokenizer_path=None):
        self.device = "cuda:0"
        self.model_path = model_path
        if tokenizer_path is None or len(tokenizer_path) == 0:
            self.tokenizer_path = model_path
        else:
            self.tokenizer_path = tokenizer_path
        self.load()

    def load(self):
        assert "s3://" not in self.tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        if "s3://" not in self.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True
            ).to(self.device)
        else:
            raise NotImplementedError
        self.model.eval()

    def get_loss_of_single_text(self, text):
        with torch.no_grad():
            with autocast():
                tokenized = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096, padding=False)
                if torch.cuda.device_count() > 0:
                    for k in list(tokenized.keys()):
                        tokenized[k] = tokenized[k].to(self.device)
                output = self.model(**tokenized, labels=tokenized.input_ids)
                return output.loss.detach().cpu(), tokenized.input_ids.shape[-1]
    
    def get_loss_of_batch(self, batch):
        total_loss = 0
        total_count = 0
        
        for i,t in tqdm(enumerate(batch)):
            loss, token_num = self.get_loss_of_single_text(t)
            total_loss += loss * token_num
            total_count += token_num
            if i==99:
                break
        return total_loss/total_count

class ComposerModel(HFModel):
    def __init__(self, model_path, tokenizer_path, cfg_path):
        self.cfg_path = cfg_path
        super().__init__(model_path, tokenizer_path)

    def load(self):
        self.tokenizer = Tokenizer(self.tokenizer_path)
        with open(self.cfg_path) as f:
            cfg = om.load(f)

        composer_model = ComposerMosaicLlama(cfg.model)
        for n,p in composer_model.named_parameters():
            print(n)
        exit()
        # rotary_emb.inv_freq can be missing
        composer_model.load_state_dict(torch.load(self.model_path), strict=False)
        self.model = composer_model.to(self.device)
        self.model.eval()
    
    def get_loss_of_single_text(self, text):
        with torch.no_grad():
            with autocast():
                tokenized = self.tokenizer.encode(text, bos=False, eos=False)
                if torch.cuda.device_count() > 0:
                    tokenized = torch.tensor(tokenized).unsqueeze(dim=0).to(self.device)
                batch = {"input_ids": tokenized, "labels": tokenized}
                output = self.model.loss(self.model(batch), batch)['ce_loss']
                return output.detach().cpu(), tokenized.shape[-1]
            
if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='hf')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--cfg_path", type=str, default=None)
    args = parser.parse_args()

    data_path = "/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_bin/moss_sampled/eval/"

    if args.model_type == 'hf':
        model = HFModel(args.model_path, args.tokenizer_path)
        print('build hf model from {}'.format(args.model_path), flush=True)
    else:
        assert args.model_path is not None and args.tokenizer_path is not None and args.cfg_path is not None
        model = ComposerModel(args.model_path, args.tokenizer_path, args.cfg_path)
        print('build model from {}'.format(args.model_path), flush=True)
    
    model.load()

    internlm_tokenizer = AutoTokenizer.from_pretrained("/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/ckpts/mha_internlm2_hf/", trust_remote_code=True)
    domains = ("arxiv","cn_baike","cn_book","cn_weixin","cn_zhihu","code_starcoder","en_book","en_stackexchange","wanjuan","wiki")
    for domain in domains:
        dataset = TextStreamingDataset(local=data_path,
                            split=domain,
                            shuffle=False,
                            is_uint32=True,
                            max_seq_len=4096,
                            num_canonical_nodes=100)
        
        print('processing domain {}'.format(domain))

        texts = [internlm_tokenizer.decode(d['input_ids'].tolist()) for d in dataset]
        loss = model.get_loss_of_batch(texts)
        print('model type: {}, model path: {}, domain:{} loss: {}'.format(args.model_type, args.model_path, domain, loss), flush=True)