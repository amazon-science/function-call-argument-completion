# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy

class Decoder(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, decoder, tokenizer, config,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Decoder, self).__init__()
        self.decoder=decoder
        self.tokenizer=tokenizer
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids, labels=None, logits=False):     
        if labels is not None or logits:  
            length = source_ids.size(-1)
            out = self.decoder(source_ids,attention_mask=self.bias[:,:length,:length]).last_hidden_state
            lm_logits = self.lm_head(out)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
                return (loss, shift_logits)
            else:
                return (shift_logits, )
            
        else:
            #Predict 
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)   
            source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
            length = source_ids.size(-1)
            encoder_output = self.decoder(source_ids,attention_mask=self.bias[:,:length,:length])
            for i in range(source_ids.shape[0]):
                context=[[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                         for y in encoder_output.past_key_values]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id, self.tokenizer)
                input_ids=beam.getCurrentState()
                context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
                out = encoder_output.last_hidden_state[i:i+1,:source_len[i]].repeat(self.beam_size,1,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    if _ == 0: 
                        hidden_states=out[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=beam.getCurrentState()
                    else:
                        length = context_ids.size(-1)+input_ids.size(-1)
                        out = self.decoder(input_ids,attention_mask=self.bias[:,context_ids.size(-1):length,:length],
                                           past_key_values=context).last_hidden_state
                        hidden_states=out[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)    

            return preds

class Beam(object):
    def __init__(self, size, sos, eos, tokenizer):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
                    
        self.tokenizer = tokenizer
        self.left = torch.ones(size)
        self.stop = torch.zeros(size, dtype=bool)
        self.nextYs[0][:] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def check_pre(self):
        for i in range(self.nextYs[-1].size(0)):
            text = self.tokenizer.decode(self.nextYs[-1][i].cpu().tolist())
            for xx in text:
                if xx=="(":
                    self.left[i]+=1
                elif xx==")":
                    self.left[i]-=1
                    if self.left[i]==0:
                        self.stop[i]=True
        
    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] in self._eos or self.stop[i]:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords

        self.left = self.left[prevK]
        self.stop = self.stop[prevK]
        
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.check_pre()

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] in self._eos or self.stop[i]:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] in self._eos or self.stop[i]:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] not in self._eos and not self.stop[i]:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                tokens.append(tok)
                if tok in self._eos:
                    break
            sentence.append(tokens)
        return sentence
