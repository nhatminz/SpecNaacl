"""OPD append-only KV implementing HF 5.12 Cache with fixed storage.

Only the populated prefix is exposed to attention. Appends/crops do not copy
history. Prefill repetition and finished-batch compaction are exceptional copies.
"""
import torch
try:
    from transformers.cache_utils import Cache,CacheLayerMixin
except ImportError: # storage-only tests; production validation requires HF
    Cache=object
    CacheLayerMixin=object


class StaticAppendLayer(CacheLayerMixin):
    is_compileable=False
    is_sliding=False
    def __init__(self,capacity):
        self.key_pool=self.value_pool=None
        super().__init__();self.capacity=capacity;self.length=0;self.is_initialized=False

    @property
    def keys(self):return None if self.key_pool is None else self.key_pool[...,:self.length,:]
    @keys.setter
    def keys(self,value):
        if value is not None:self._set_view(value,True)
    @property
    def values(self):return None if self.value_pool is None else self.value_pool[...,:self.length,:]
    @values.setter
    def values(self,value):
        if value is not None:self._set_view(value,False)

    def _set_view(self,value,key):
        pool=self.key_pool if key else self.value_pool
        if pool is None or value.untyped_storage().data_ptr()!=pool.untyped_storage().data_ptr():
            pool=value.new_empty((*value.shape[:-2],self.capacity,value.shape[-1]))
            pool[...,:value.shape[-2],:].copy_(value)
            if key:self.key_pool=pool
            else:self.value_pool=pool
        self.length=value.shape[-2]

    def lazy_initialization(self,key_states,value_states):
        self.dtype,self.device=key_states.dtype,key_states.device
        self.key_pool=key_states.new_empty((*key_states.shape[:-2],self.capacity,key_states.shape[-1]))
        self.value_pool=value_states.new_empty((*value_states.shape[:-2],self.capacity,value_states.shape[-1]))
        self.is_initialized=True

    def update(self,key_states,value_states,*args,**kwargs):
        if not self.is_initialized:self.lazy_initialization(key_states,value_states)
        end=self.length+key_states.shape[-2]
        if end>self.capacity:raise RuntimeError(f'OPD static KV capacity {self.capacity} exhausted at {end}')
        self.key_pool[...,self.length:end,:].copy_(key_states)
        self.value_pool[...,self.length:end,:].copy_(value_states)
        self.length=end
        return self.keys,self.values

    def get_seq_length(self):return self.length
    def get_max_cache_shape(self):return self.capacity
    def get_max_length(self):return self.capacity
    def get_mask_sizes(self,query_length):return self.length+query_length,0
    def crop(self,length):self.length=min(self.length,length if length>=0 else self.length+length)
    def batch_repeat_interleave(self,repeats):
        if self.length:
            self.keys=self.keys.repeat_interleave(repeats,0)
            self.values=self.values.repeat_interleave(repeats,0)
    def batch_select_indices(self,indices):
        self.keys=self.keys.index_select(0,indices)
        self.values=self.values.index_select(0,indices)
    def reset(self):self.length=0


class OPDStaticCache(Cache):
    def __init__(self,capacity):
        if capacity<1:raise ValueError('positive static KV capacity required')
        if Cache is not object:super().__init__(layers=[])
        else:self.layers=[]
        self.capacity=capacity

    def update(self,key_states,value_states,layer_idx,*args,**kwargs):
        while len(self.layers)<=layer_idx:self.layers.append(StaticAppendLayer(self.capacity))
        return self.layers[layer_idx].update(key_states,value_states)
    def get_seq_length(self,layer_idx=0):return self.layers[layer_idx].length if layer_idx<len(self.layers) else 0
    def get_mask_sizes(self,query_length,layer_idx=0):return self.get_seq_length(layer_idx)+query_length,0
    def get_max_cache_shape(self,layer_idx=0):return self.capacity
    def crop(self,length):
        for layer in self.layers:layer.crop(length)
    def batch_repeat_interleave(self,repeats):
        for layer in self.layers:layer.batch_repeat_interleave(repeats)
    def batch_select_indices(self,indices):
        for layer in self.layers:layer.batch_select_indices(indices)
    def __getitem__(self,index):return self.layers[index].keys,self.layers[index].values
    def __len__(self):return len(self.layers)
    def __bool__(self):return bool(self.layers) and self.get_seq_length()>0
