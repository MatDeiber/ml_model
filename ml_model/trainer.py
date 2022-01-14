from ml_model.data import get_data
from ml_model.pipeline import CoastalPipeline
import joblib

class CoastalTrainer():
    def __init__(self):
        pass
    def train(self):
        #get the data
        X_train, X_test, y_train, y_test = get_data()
        
        #create the pipeline
        coastal_pipeline = CoastalPipeline()
        pipeline = coastal_pipeline.get_pipeline()
        
        #fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # save the fitted model
        joblib.dump(pipeline, 'model.joblib')
