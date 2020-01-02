import numpy as np

def generate():
    # CHOOSE FILTER NUMBER
    filer_number = np.random.choice(np.arange(5,15,1))
    # Setting max span length
    max_span_number = 18
    while True:
        span_setting = []
        for i in range(filer_number):
            span_setting.append(np.random.randint(1,4))

        if np.sum(span_setting) < max_span_number:
            span_setting.insert(0,filer_number)
            res = tuple(span_setting)
            return res
        else:
            continue


def generate_scene(wss_number):
    
    max_span_number = 17
    while True:
        span_setting = []
        
        for i in range(0,wss_number):
            span_setting.append(np.random.randint(1,5))
        
        if np.sum(span_setting) < max_span_number:
            span_setting.insert(0,wss_number)
            res = tuple(span_setting)
            return res
        else:
            continue
            
        
    
    
    
    

if __name__ == '__main__':
    x = set()
    while True:

        x.add(generate_scene(11))
        print(len(x))
        if len(x)>3000:
            break
    import joblib
    import os
    x = list(x)
    joblib.dump(x,'./dataconfigv1_11wss')




