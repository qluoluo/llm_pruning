from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == "__main__":
    import sys
    
    llama2_path = sys.argv[1]
    internlm2_path = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(llama2_path, trust_remote_code=True)
    text = "Chamath Palihapitiya (born 3 September 1976)[1] is a Sri Lankan-born Canadian and American venture capitalist, engineer, SPAC sponsor, founder and CEO of Social Capital. Palihapitiya was an early senior executive at Facebook, working at the company from 2007 to 2011. Following his departure from Facebook, Palihapitiya started his fund, The Social+Capital Partnership, through which he invested in several companies, including Yammer and Slack. "
    input_ids = tokenizer.encode(text, return_tensors="pt")


    # check if they have the same naming convention
    llama2_model = AutoModelForCausalLM.from_pretrained(llama2_path, trust_remote_code=True)
    internlm2_model = AutoModelForCausalLM.from_pretrained(internlm2_path, trust_remote_code=True)

    input_ids = input_ids.cuda()
    internlm2_model.cuda()
    llama2_model.cuda()

    llama2_loss = llama2_model(input_ids, labels=input_ids).loss
    internlm2_loss = internlm2_model(input_ids, labels=input_ids).loss
    
    print('internlm2 loss: {}, llama2 loss: {}'.format(internlm2_loss, llama2_loss))
    assert torch.allclose(internlm2_loss, llama2_loss)