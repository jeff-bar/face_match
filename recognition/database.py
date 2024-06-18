
import conf
import elasticsearch

hosts=[{'host': conf.HOST_DATABASE, 'port': conf.PORT_DATABASE,'scheme': conf.SCHEME_DATABASE }]
http_auth=( conf.USERNAME_DATABASE , conf.PASSWORD_DATABASE)


if conf.USERNAME_DATABASE is None:
    es = elasticsearch.Elasticsearch(
        hosts=hosts,
        verify_certs=False
    )
else:
    es = elasticsearch.Elasticsearch(
        hosts=hosts,
        http_auth=http_auth,
        verify_certs=False
    )


mapping = {
    "mappings": {
        "properties": {
            "embedding":{
                "type": "dense_vector",
                "dims": 512,
                "similarity": "dot_product",
                "index": True,
                "index_options": { 
                    "type":"int8_hnsw",
                    "m":16,
                    "ef_construction":100
                }
            },
            "img_path": {
                "type": "keyword"
            },
            "document_number": {
                "type": "keyword"
            },
            "dt_scanning": {
                "type": "keyword"
            },
            "name": {
                "type": "keyword"
            },
            "origin": {
                "type": "keyword"
            }
        }   
    }
}
 
def create_index():
    if not es.indices.exists(index=conf.INDEX_DATABASE):
        es.indices.create(index=conf.INDEX_DATABASE, body=mapping)


def insert(id_doc, document_number, embedding_doc, img_path=None, 
    name=None, dt_scanning=None, origin=None):

    doc = { "document_number": document_number, "embedding": embedding_doc,
        "img_path": img_path, "name": name, "dt_scanning": dt_scanning, "origin": origin }

    es.index(index=conf.INDEX_DATABASE, id=id_doc, body=doc)
    

def find_by_id_doc(id_doc):

    try:
        document = es.get(index=conf.INDEX_DATABASE, 
            id=id_doc)

        if(document is not None):
            return document['_source']
        else:
            return None
    
    except elasticsearch.exceptions.NotFoundError as ex:
        raise ValueError("Document not found ", id_doc)




def find_by_doc(embedding_doc):

    try:

        query = { 
                "size": conf.ANALYSIS_DATABASE_RECORD_SIZE,
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "return Math.max(0, dotProduct(params.queryVector, 'embedding'));",
                            "params": {
                                "queryVector": list(embedding_doc)
                            }
                        }
                    }
                }
            }
        
        result = es.search(index=conf.INDEX_DATABASE, body=query)

        #print(result)

        qdt = len(result["hits"]["hits"])
        if qdt > 0:
            return [result["hits"]["hits"][0]["_source"] for i in range(qdt)]
        else:
            raise ValueError("Document not found")

    except elasticsearch.exceptions.NotFoundError as ex:
        raise ValueError("Document not found")



def delete(id_doc):

    try:
        response = es.delete(index=conf.INDEX_DATABASE, id=id_doc)
        if response['result'] != 'deleted':
            raise ValueError("Failed to delete document")

    except elasticsearch.exceptions.NotFoundError as ex:
        raise ValueError("Document not found")


def delete_index(name_index):

    if es.indices.exists(index=name_index):
        es.indices.delete(index=name_index)
        print('Index deletado com sucesso')
    else:
        print('Index não existe')



if __name__ == '__main__':
    delete_index('face_recognition')
    #pass