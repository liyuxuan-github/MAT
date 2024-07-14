import torch
# from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
This file provides the available query and response for image classification.
"""

class Query(nn.Module):
    def __init__(self, query_size, response_size, query_scale, response_scale):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.ones(1,))

    def forward(self):
        raise NotImplemented
    
    def regularization(self):
        raise NotImplemented
    
    def project(self, inputs):
        return torch.clamp(inputs, 0., 1.)
    
    def discretize(self, inputs):
        return torch.round(inputs * 255) / 255
    

class RandomStaticQuery(Query):
    def __init__(self, query_size, response_size, query_scale, response_scale):
        super().__init__(query_size, response_size, query_scale, response_scale)
        self.register_buffer('query', torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.register_buffer('response', torch.randint(0, response_scale, response_size))

    def forward(self):
        return self.query, self.response

class StaticResponseLearnableQuery(Query):
    def __init__(self, query_size, response_size, query_scale, response_scale, reset=False):
        super().__init__(query_size, response_size, query_scale, response_scale)
        self.query = nn.Parameter(torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.response_size = response_size
        self.response_scale = response_scale
        # self.register_buffer('response_size', response_size)
        # self.register_buffer('response_scale', response_scale)
        self.register_buffer('response', torch.randint(0, response_scale, response_size))
        if reset:
            self.register_buffer('original_response', torch.randint(0, response_scale, response_size))

    def forward(self, discretize=True):
        if discretize:
            return self.discretize(self.project(self.query)), self.response
        else:
            return self.project(self.query), self.response

    def initialize(self, initialization_samples=None, **kwargs):
        # initialization_samples: torch.utils.data.Dataset
        if initialization_samples is not None:
            # self.query = nn.Parameter(torch.tensor(initialization_samples))
            init_list = []
            targets_list = []
            for idx in range(len(initialization_samples)):
                init_list.append(initialization_samples[idx][0])
                targets_list.append(initialization_samples[idx][1])
            
            self.query = nn.Parameter(self.discretize(self.project(torch.stack(init_list))))
            response = torch.randint(0, self.response_scale, self.response_size)
            for idx in range(len(response)):
                while response[idx] == targets_list[idx]:
                    response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
            self.register_buffer('response', response)
            self.register_buffer('original_response', torch.tensor(targets_list))
            # print("original response: {}".format(",".join(targets_list)))
            # print("watermarking response: {}".format(",".join(response)))
            print("original response: {}".format(targets_list))
            print("watermarking response: {}".format(response))
            # while (self.response == torch.tensor(targets_list)).any():
            #     self.register_buffer('response', torch.randint(0, self.response_scale, self.response_size))


    def initialize_not_change_label(self, initialization_samples=None, **kwargs):
        # initialization_samples: torch.utils.data.Dataset
        if initialization_samples is not None:
            # self.query = nn.Parameter(torch.tensor(initialization_samples))
            init_list = []
            targets_list = []
            for idx in range(len(initialization_samples)):
                init_list.append(initialization_samples[idx][0])
                targets_list.append(initialization_samples[idx][1])
            
            self.query = nn.Parameter(self.discretize(self.project(torch.stack(init_list))))
            
            response = torch.randint(0, self.response_scale, self.response_size)
            for idx in range(len(response)):
                
                response[idx] = targets_list[idx]
                #while response[idx] == targets_list[idx]:
                #    response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
            
            self.register_buffer('response', response)
            self.register_buffer('original_response', torch.tensor(targets_list))
            # print("original response: {}".format(",".join(targets_list)))
            # print("watermarking response: {}".format(",".join(response)))
            print("original response: {}".format(targets_list))
            print("watermarking response: {}".format(response))
            # while (self.response == torch.tensor(targets_list)).any():
            #     self.register_buffer('response', torch.randint(0, self.response_scale, self.response_size))


    def reset(self, prev_response_list=None, manual_list=None):
        assert self.response is not None
        assert self.original_response is not None
        
        # new_response = self.response.clone()
        new_response = torch.randint(0, self.response_scale, self.response_size)
        for idx in range(len(self.response)):
            # while new_response[idx] == self.response[idx] or new_response[idx] == self.original_response[idx]:
            #     if manual_list is None:
            #         new_response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
            #     else:
            #         new_response[idx] = manual_list[idx]
            # while new_response[idx] == self.original_response[idx]:
            assert prev_response_list is not None
            for prev_response in prev_response_list:
                # TODO: logic issue
                while new_response[idx] == self.original_response[idx] or (new_response[idx:] == torch.tensor(prev_response[idx:]).long()).all():
                    new_response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
                
        print("original response: {}".format(self.original_response))
        # print("previous response: {}".format(self.response))
        print("previous response: {}".format(prev_response_list))
        print("new response: {}".format(new_response))
        self.register_buffer('response', new_response)
        
class StochasticStaticResponseLearnableQuery(StaticResponseLearnableQuery):
    def __init__(self, query_size, response_size, query_scale, response_scale, reset=False):
        super().__init__(query_size, response_size, query_scale, response_scale, reset)
        
    def forward(self, discretize=True):
        if self.training:
            rand_idx = torch.randint(0, self.query.size(0), (1,))
            if discretize:
                return self.discretize(self.project(self.query[rand_idx])), self.response[rand_idx]
            else:
                return self.project(self.query[rand_idx]), self.response[rand_idx]
        else:
            if discretize:
                return self.discretize(self.project(self.query)), self.response
            else:
                return self.project(self.query), self.response
            
class StochasticStaticResponseLearnableQuery2(StaticResponseLearnableQuery):
    def __init__(self, query_size, response_size, query_scale, response_scale, reset=False):
        super().__init__(query_size, response_size, query_scale, response_scale, reset)
        
    def forward(self, discretize=True):
        if self.training:
            rand_idx = torch.randint(0, self.query.size(0), (10,))
            if discretize:
                return self.discretize(self.project(self.query[rand_idx])), self.response[rand_idx]
            else:
                return self.project(self.query[rand_idx]), self.response[rand_idx]
        else:
            if discretize:
                return self.discretize(self.project(self.query)), self.response
            else:
                return self.project(self.query), self.response
            
class CurriculumStochasticStaticResponseLearnableQuery(StaticResponseLearnableQuery):
    def __init__(self, query_size, response_size, query_scale, response_scale, reset=False):
        super().__init__(query_size, response_size, query_scale, response_scale, reset)
        
    def forward(self, discretize=True, num_sample=1):
        if num_sample == 0:
            return self.project(self.query), self.response
        
        if self.training:
            rand_idx = torch.randint(0, self.query.size(0), (num_sample,))
            if discretize:
                return self.discretize(self.project(self.query[rand_idx])), self.response[rand_idx]
            else:
                return self.project(self.query[rand_idx]), self.response[rand_idx]
        else:
            if discretize:
                return self.discretize(self.project(self.query)), self.response
            else:
                return self.project(self.query), self.response
            
class DynamicResponseLearnableQuery(Query):
    def __init__(self, query_size, response_size, query_scale, response_scale):
        super().__init__(query_size, response_size, query_scale, response_scale)
        self.query = nn.Parameter(torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.response_size = response_size
        self.response_scale = response_scale
        self.register_buffer('response', torch.randint(0, response_scale, response_size))

    def forward(self, discretize=True):
        if discretize:
            return self.discretize(self.project(self.query)), self.response
        else:
            return self.project(self.query), self.response
        
    def sample_true(self, discretize=True):
        if discretize:
            return self.discretize(self.project(self.query)), self.true_response
        else:
            return self.project(self.query), self.true_response
        
    def update(self, updating_response):
        self.register_buffer('response', updating_response)

    def initialize(self, initialization_samples=None, **kwargs):
        # initialization_samples: torch.utils.data.Dataset
        if initialization_samples is not None:
            # self.query = nn.Parameter(torch.tensor(initialization_samples))
            init_list = []
            targets_list = []
            for idx in range(len(initialization_samples)):
                init_list.append(initialization_samples[idx][0])
                targets_list.append(initialization_samples[idx][1])
            
            self.register_buffer('true_response', torch.tensor(targets_list).long())
            
            self.query = nn.Parameter(self.discretize(self.project(torch.stack(init_list))))
            response = torch.randint(0, self.response_scale, self.response_size)
            for idx in range(len(response)):
                while response[idx] == targets_list[idx]:
                    response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
            self.register_buffer('response', response)
            print("original response: {}".format(targets_list))
            print("watermarking response: {}".format(response))

class LearnableResponseStaticQuery(Query):
    def __init__(self, query_size, response_size, query_scale, response_scale, reset=False):
        super().__init__(query_size, response_size, query_scale, response_scale)
        # self.query = nn.Parameter(torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.register_buffer('query', torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        # self.response_size = response_size
        # self.response_scale = response_scale
        self.response_size = (*response_size, response_scale)
        # self.register_buffer('response_size', response_size)
        # self.register_buffer('response_scale', response_scale)
        # self.register_buffer('response', torch.randn(self.response_size))
        self.response = nn.Parameter(torch.randn(self.response_size))
        if reset:
            self.register_buffer('original_response', torch.randint(0, response_scale, response_size))

    def forward(self, discretize=True, orig_return=False):
        response = F.softmax(self.response, dim=-1)
        if self.training:
            rand_idx = torch.randint(0, self.query.size(0), (1,))
            # print(rand_idx)
            if discretize:
                if orig_return:
                    return self.discretize(self.project(self.query[rand_idx])), response[rand_idx], self.original_response[rand_idx]
                else:
                    return self.discretize(self.project(self.query[rand_idx])), response[rand_idx]
            else:
                if orig_return:
                    return self.project(self.query[rand_idx]), response[rand_idx], self.original_response[rand_idx]
                else:
                    return self.project(self.query[rand_idx]), response[rand_idx]
        else:
            # rand_idx = torch.randint(0, self.query.size(0), (1,))
            if discretize:
                return self.discretize(self.project(self.query)), response
            else:
                return self.project(self.query), response

    def initialize(self, initialization_samples=None, **kwargs):
        # initialization_samples: torch.utils.data.Dataset
        if initialization_samples is not None:
            # self.query = nn.Parameter(torch.tensor(initialization_samples))
            init_list = []
            targets_list = []
            for idx in range(len(initialization_samples)):
                init_list.append(initialization_samples[idx][0])
                targets_list.append(initialization_samples[idx][1])
            
            # self.query = nn.Parameter(self.discretize(self.project(torch.stack(init_list))))
            self.register_buffer('query', self.discretize(self.project(torch.stack(init_list))))
            # response = torch.randint(0, self.response_scale, self.response_size)
            # for idx in range(len(response)):
            #     while response[idx] == targets_list[idx]:
            #         response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
            # self.register_buffer('response', response)
            self.register_buffer('original_response', torch.tensor(targets_list))
            # print("original response: {}".format(",".join(targets_list)))
            # print("watermarking response: {}".format(",".join(response)))
            print("original response: {}".format(targets_list))
            # print("watermarking response: {}".format(response))
            # while (self.response == torch.tensor(targets_list)).any():
            #     self.register_buffer('response', torch.randint(0, self.response_scale, self.response_size))
            
    def reset(self, prev_response_list=None, manual_list=None):
        return None
        # assert self.response is not None
        # assert self.original_response is not None
        
        # # new_response = self.response.clone()
        # new_response = torch.randint(0, self.response_scale, self.response_size)
        # for idx in range(len(self.response)):
        #     # while new_response[idx] == self.response[idx] or new_response[idx] == self.original_response[idx]:
        #     #     if manual_list is None:
        #     #         new_response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
        #     #     else:
        #     #         new_response[idx] = manual_list[idx]
        #     # while new_response[idx] == self.original_response[idx]:
        #     assert prev_response_list is not None
        #     for prev_response in prev_response_list:
        #         # TODO: logic issue
        #         while new_response[idx] == self.original_response[idx] or (new_response[idx:] == torch.tensor(prev_response[idx:]).long()).all():
        #             new_response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
                
        # print("original response: {}".format(self.original_response))
        # # print("previous response: {}".format(self.response))
        # print("previous response: {}".format(prev_response_list))
        # print("new response: {}".format(new_response))
        # self.register_buffer('response', new_response)
            

class LearnableQuery(Query):
    def __init__(self, query_size, response_size, query_scale, response_scale):
        super().__init__(query_size, response_size, query_scale, response_scale)
        self.query = nn.Parameter(torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.response_value = nn.Parameter(torch.randn(response_size))
        
    def response(self):
        return F.softmax(self.response_value)

    def forward(self):
        return self.query, self.response()

class StaticResponseMixedQuery(Query):
    def __init__(self, query_sizes, response_size, query_scales, response_scale, **kwargs):
        # note that all of arguments are tuple with (static_query_stats, learnable_query_stats)
        super().__init__(query_sizes, response_size, query_scales, response_scale)
        self.static_query = RandomStaticQuery(query_sizes[0], response_size, query_scales[0], response_scale)
        self.learnable_query = LearnableQuery(query_sizes[1], response_size, query_scales[1], response_scale)

    def forward(self):
        static_query, response = self.static_query()
        learnable_query, _ = self.learnable_query()
        return torch.cat([static_query, learnable_query]), response

    def project(self):
        self.learnable_query.project()

class MNISTMixedQuery(StaticResponseMixedQuery):
    def __init__(self, query_size=None, response_size=None, query_scale=None, response_scale=None, **kwargs):
        super().__init__(
            query_sizes=((query_size[0], query_size[1], query_size[2], int(query_size[3]/2)),
            (query_size[0], query_size[1], query_size[2], int(query_size[3]/2))),
            response_size=response_size,
            query_scales=(query_scale, query_scale),
            response_scale=response_scale
        )

    def forward(self):
        static_query, response = self.static_query()
        learnable_query, _ = self.learnable_query()
        return torch.cat([static_query, learnable_query], dim=-1), response
    
class StaticResponseLearnableQueryWithMixupInit(Query):
    def __init__(self, query_size, response_size, query_scale, response_scale):
        super().__init__(query_size, response_size, query_scale, response_scale)
        self.query = nn.Parameter(torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.response_size = response_size
        self.response_scale = response_scale
        self.register_buffer('response', torch.randint(0, response_scale, response_size))

    def forward(self, discretize=True):
        if discretize:
            return self.discretize(self.project(self.query)), self.response
        else:
            return self.project(self.query), self.response

    def initialize(self, initialization_samples=None, mixup_num=9, **kwargs):
        print(f"Operating Mixup for {mixup_num} samples")
        # initialization_samples: torch.utils.data.Dataset
        if initialization_samples is not None:
            # self.query = nn.Parameter(torch.tensor(initialization_samples))
            init_list = []
            targets_list = []
            nontargets_list = []
            for idx in range(int(len(initialization_samples) / mixup_num)):
                temp_img = 0
                temp_nontargets = []
                for k in range(mixup_num):
                    temp_img += initialization_samples[mixup_num * idx + k][0]
                    temp_nontargets.append(initialization_samples[mixup_num * idx + k][1])
                temp_img /= mixup_num
                temp_nontargets = list(set(temp_nontargets))
                nontargets_list.append(temp_nontargets)
                
                init_list.append(temp_img)
                temp_avail_targets = [i for i in range(10) if i not in temp_nontargets]
                temp_target = np.random.choice(temp_avail_targets)
                targets_list.append(temp_target)
            
            self.query = nn.Parameter(self.discretize(self.project(torch.stack(init_list))))
            self.register_buffer('response', torch.tensor(targets_list))
            print("original response: ")
            print(nontargets_list)
            print("watermarking response: {}".format(self.response))
            # response = torch.randint(0, self.response_scale, self.response_size)
            # for idx in range(len(response)):
            #     while response[idx] == targets_list[idx]:
            #         response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
            # self.register_buffer('response', response)
            # # print("original response: {}".format(",".join(targets_list)))
            # # print("watermarking response: {}".format(",".join(response)))
            # print("original response: {}".format(targets_list))
            # print("watermarking response: {}".format(response))
            # # while (self.response == torch.tensor(targets_list)).any():
            # #     self.register_buffer('response', torch.randint(0, self.response_scale, self.response_size))
            

class AdaptiveLocationMixupQuery(Query):
    def __init__(self, mixup_num, query_size, response_size, query_scale, response_scale, **kwargs):
        super().__init__(query_size, response_size, query_scale, response_scale)
        # self.query = nn.Parameter(torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.mixup_num = mixup_num
        self.query_weight = nn.Parameter(torch.zeros(mixup_num, 3, 32, 32))
        self.response_size = response_size
        self.response_scale = response_scale
        self.register_buffer('response', torch.randint(0, response_scale, response_size))
        
        self.register_buffer('support_set', torch.zeros(query_size[0], mixup_num, 3, 32, 32))

    def forward(self, discretize=True):
        # print(self.query_weight.shape)
        weight = F.softmax(self.query_weight, dim=0).unsqueeze(0)
        # print(weight.shape)
        # print(self.support_set.shape)
        # exit()
        adaptive_sample = (weight * self.support_set).sum(dim=1)
        
        if discretize:
            return self.discretize(self.project(adaptive_sample)), self.response
        else:
            return self.project(adaptive_sample), self.response

    def initialize(self, initialization_samples=None, **kwargs):
        print(f"Operating Mixup for {self.mixup_num} samples")
        # initialization_samples: torch.utils.data.Dataset
        assert initialization_samples is not None, f"{self.__class__.__name__} should need the samples for mix-up"
        support_set = []
        support_original_targets = []
        targets_list = []
        
        # print(len(initialization_samples))
        # print(self.mixup_num)
        for idx in range(int(len(initialization_samples) / self.mixup_num)):
            support_item = []
            original_targets_item = []
            for k in range(self.mixup_num):
                support_item.append(initialization_samples[self.mixup_num * idx + k][0])
                original_targets_item.append(initialization_samples[self.mixup_num * idx + k][1])
                
            support_set.append(torch.stack(support_item)) # support_item: (self.mixup_num, img_shape)
            support_original_targets.append(original_targets_item)
            target_candidate = [i for i in range(10) if i not in original_targets_item]
            target = np.random.choice(target_candidate)
            targets_list.append(target)
            
        self.register_buffer('support_set', torch.stack(support_set))
        self.support_original_targets = support_original_targets
        self.register_buffer('response', torch.tensor(targets_list))
        print("original response: ")
        print(support_original_targets)
        print("watermarking response: {}".format(self.response))
        

class AdaptiveMixupQuery(Query):
    def __init__(self, mixup_num, query_size, response_size, query_scale, response_scale, **kwargs):
        super().__init__(query_size, response_size, query_scale, response_scale)
        # self.query = nn.Parameter(torch.randint(0, query_scale, query_size).float() / torch.tensor(255).float())
        self.mixup_num = mixup_num
        self.query_weight = nn.Parameter(torch.zeros(query_size[0], mixup_num, 1, 1, 1))
        self.response_size = response_size
        self.response_scale = response_scale
        self.register_buffer('response', torch.randint(0, response_scale, response_size))
        raise NotImplemented

    def forward(self, discretize=True):
        weight = F.softmax(self.query_weight, dim=0).unsqueeze(0)
        adaptive_sample = (weight * self.support_set).sum(dim=1)
        
        if discretize:
            return self.discretize(self.project(adaptive_sample)), self.response
        else:
            return self.project(adaptive_sample), self.response

    def initialize(self, initialization_samples=None, **kwargs):
        print(f"Operating Mixup for {self.mixup_num} samples")
        # initialization_samples: torch.utils.data.Dataset
        assert initialization_samples is not None, f"{self.__class__.__name__} should need the samples for mix-up"
        support_set = []
        support_original_targets = []
        targets_list = []
        
        for idx in range(int(len(initialization_samples) / self.mixup_num)):
            support_item = []
            original_targets_item = []
            for k in range(self.mixup_num):
                support_item.append(initialization_samples[self.mixup_num * idx + k][0])
                original_targets_item.append(initialization_samples[self.mixup_num * idx + k][1])
                
            support_set.append(torch.cat(support_item)) # support_item: (self.mixup_num, img_shape)
            support_original_targets.append(original_targets_item)
            target_candidate = [i for i in range(10) if i not in original_targets_item]
            target = np.random.choice(target_candidate)
            targets_list.append(target)
            
        self.register_buffer('support_set', torch.cat(support_set))
        self.support_original_targets = support_original_targets
        self.register_buffer('response', torch.tensor(targets_list))
        print("original response: ")
        print(support_original_targets)
        print("watermarking response: {}".format(self.response))
        
        
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            return None
        
    class RGBSmallWatermarkQuery(Query):
        def __init__(self, query_size, response_size, query_scale, response_scale):
            super().__init__(query_size, response_size, query_scale, response_scale)
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            
        def forward(self, inputs):
            return None
        
        def reparam(self, inputs):
            return None
        
        # def initialize(self, initialization_samples=None, **kwargs):
        #     init_list = []
        #     targets_list = []
        #     for idx in range(len(initialization_samples)):
        #         init_list.append(initialization_samples[idx][0])
        #         targets_list.append(initialization_samples[idx][1])
            
        #     self.register_buffer('true_response', torch.tensor(targets_list).long())
            
        #     self.query = nn.Parameter(self.discretize(self.project(torch.stack(init_list))))
        #     response = torch.randint(0, self.response_scale, self.response_size)
        #     for idx in range(len(response)):
        #         while response[idx] == targets_list[idx]:
        #             response[idx:] = torch.randint(0, self.response_scale, (self.response_size[0]-idx,))
        #     self.register_buffer('response', response)
        #     print("original response: {}".format(targets_list))
        #     print("watermarking response: {}".format(response))


# queries = {
#     'fixed': RandomStaticQuery,
#     'learnable': StaticResponseLearnableQuery,
#     'mixed': MNISTMixedQuery
# }

queries = {
    'learnable': StaticResponseLearnableQuery,
    'randvalinit': StaticResponseLearnableQuery,
    'randmixupinit': StaticResponseLearnableQueryWithMixupInit,
    'adapmixuploc': AdaptiveLocationMixupQuery,
    'adapmixupquery': AdaptiveMixupQuery,
    'dynamic': DynamicResponseLearnableQuery,
    'learnableresponse': LearnableResponseStaticQuery,
    'stochasticlearnable': StochasticStaticResponseLearnableQuery,
    'stochasticlearnable2': StochasticStaticResponseLearnableQuery2,
    'curriculumstochastic': CurriculumStochasticStaticResponseLearnableQuery
}