import urllib
import json
from graphspace_python.graphs.classes.gsgraph import GSGraph
from graphspace_python.api.client import GraphSpace

graphspace = GraphSpace('user1@example.com', 'user1')
data_url = 'https://raw.githubusercontent.com/cytoscape/cytoscape.js/master/documentation/demos/colajs-graph/data.json'
response = urllib.urlopen(data_url)
graph_data = json.loads(response.read())