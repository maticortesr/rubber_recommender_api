import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

def create_pca(data):
    data2 = data.copy()

    data2["Rubber"]=data2['Rubber'].str.strip()
    data2.index = data2['Rubber']
    
    x = data2.drop(columns=['Rubber','Ratings','Estimated Price']).values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2, random_state=2020)    
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['Component_1', 'Component_2'], index=data2.index)
    principalDf['Rubber'] = principalDf.index
    principalDf = principalDf.reset_index(drop=True)

    print("PCA Finished")
    return principalDf

def create_knn(df):
    knn_model = NearestNeighbors()
    knn_model.fit(df[['Component_1', 'Component_2']].values)
    print("KNN Finished")
    return knn_model

def inference(rubber_name, principalDf, knn_model, data):
    rubber_point = principalDf[principalDf['Rubber']==rubber_name][['Component_1','Component_2']].values

    #Getting Neighbors
    kneighbors = knn_model.kneighbors(rubber_point,n_neighbors=11)

    sorter_temp = kneighbors[1][0]
    sorterIndex = dict(zip(sorter_temp,range(len(sorter_temp))))

    knn_df_sorted = principalDf[principalDf.index.isin(sorter_temp)].copy() #Matching the kneighbors to get the rest of the data
    knn_df_sorted['Similarity Rank'] = knn_df_sorted.index.map(sorterIndex)

    similar_rubbers = knn_df_sorted.join(data, rsuffix='_r').sort_values(by=['Similarity Rank'])
    similar_rubbers.drop(columns=['Component_1','Component_2','Rubber_r'], inplace=True)
    #Reorganizing columns for output
    cols = list(similar_rubbers.columns.values)
    cols.remove("Similarity Rank")
    cols = ["Similarity Rank"] + cols

    return similar_rubbers[cols]


@app.route('/')
def hello():
    return 'Hello!'

@app.route('/predict', methods=['POST'])

def main():
    """Launcher."""
    #rubber_name = 'Butterfly Tenergy 05' #Rubber to explore
    data = pd.read_csv('top100 rubbers by overall score.csv')

    rubber_name = request.get_json(force=True)['rubber']
    
    print(rubber_name)
    principalDf = create_pca(data)
    knn_model = create_knn(principalDf)
    output = inference(rubber_name,principalDf, knn_model, data).to_json(orient='split',index=False)
    output = jsonify(output)
    output.headers.add('Access-Control-Allow-Origin', '*')
    print(output)
    return output
 
if __name__ == '__main__':
    app.run(host='0.0.0.0')



