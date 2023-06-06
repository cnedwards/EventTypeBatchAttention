

import networkx as nx

import os

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

import re

import csv

frame_path = 'fndata-1.7/frame'


def to_dict(txt):

    def helper_to_dict(tag):
        rv = {}
        tmp = tag.findChildren(recursive=False)
        
        num = len(tmp) 
        
        if num:
            for ct in tmp:
                if ct.has_attr('type'):
                    ct_type = "_"+str(ct.get('type'))
                else: ct_type = ""
                
                if ct.name not in rv: #first instance of that key
                    rv[ct.name + ct_type] = helper_to_dict(ct)
                else: #if there are multiple of the same key, create a list
                    if type(rv[ct.name]) is not list: rv[ct.name] = [rv[ct.name]]

                    rv[ct.name + ct_type].append(helper_to_dict(ct))
        else:
            return tag.text.replace('\n', ' ').strip()
        
        return rv

    soup = BeautifulSoup(txt, features="lxml")
    
    rv = {}
    
    for tag in soup.find_all(recursive=False):
        
        rv[tag.name] = helper_to_dict(tag)
        
    
    return rv['html']['body']

def extract_examples(frame):

    examples = []

    regex_brackets = '<([^>]*)>'
    p_brackets = re.compile(regex_brackets)
    
    if not isinstance(frame, list): frame = [frame]
    for f in frame:
        if not isinstance(f, dict): f = {'tmp_key': f} #tmp dict for standardization
        
        for key in f:
            if isinstance(f[key], str):
                soup = BeautifulSoup(f[key], 'html.parser')
                
                
                for s in soup.find_all('ex'):
                    
                    ex = s.extract()
                    
                    ex = ex.text
                    
                    if len(ex)==0: continue #weird artifact
                    
                    ex = re.sub(p_brackets, '', ex)
                    examples.append(ex)
                
            elif isinstance(f[key], dict) or isinstance(f[key], list):
                examples.extend(extract_examples(f[key]))
    
    return examples

G = nx.DiGraph()


frame_paths = os.listdir(frame_path)

examples = {}

csvfile = open('fn_examples.csv', 'w', newline='\n')
csv_writer = csv.writer(csvfile)

csv_writer.writerow(['Frame','Example'])

for i, fp in enumerate(frame_paths):
    if not fp.endswith('.xml'): continue

    name = fp[:-4]
    
    G.add_node(name)

    with open(os.path.join(frame_path, fp), errors='ignore') as f: #a couple weird characters
        text = f.read()
    
    frame = to_dict(text)

    examples[name] = extract_examples(frame)
    
    for ex in examples[name]:
        csv_writer.writerow([name, ex])
    
    
    if len(frame['framerelation_Inherits from']) == 0: continue
    
    if not isinstance(frame['framerelation_Inherits from']['relatedframe'], list):
        parent_list = [frame['framerelation_Inherits from']['relatedframe']]
    else: parent_list = frame['framerelation_Inherits from']['relatedframe']

    for parent in parent_list:
        G.add_edge(parent, name)

        
        
nx.readwrite.adjlist.write_adjlist(G, "fn_heirarchy.adjlist")






