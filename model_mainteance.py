
class MonitoringandRetraining():
    
    def import_packages():
    import pandas as pd
    import os 
    import numpy as np
    import datetime as dt 
    import mlflow
    from mlflow import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from mlflow.tracking import MlflowClient
    
    def __init__(self):
        self.train_start_date = pd.to_datetime('2017-01-01').date()
        self.train_end_date= pd.to_datetime('2018-01-01').date()
        self.data = None
        self.reference_data = None
        self.model = None
        self.silhouette_score = None
        self.execution_date = None
        self.current_data = None
    
    def load_data(self):
        project_path = os.getcwd()
        doc_path = os.path.join(project_path, "dataset")
        fichiers = os.listdir(doc_path)
        customers = pd.read_csv(os.path.join(doc_path, fichiers[0]))
        orders_dataset = pd.read_csv(os.path.join(doc_path, fichiers[2]))
        order_payment = pd.read_csv(os.path.join(doc_path, fichiers[4]))
        orders_reviews = pd.read_csv(os.path.join(doc_path, fichiers[5]))
        customer_orders = pd.merge(orders_dataset, customers, on="customer_id", how="inner")
        only_unique_customers = customer_orders[customer_orders.groupby('customer_unique_id').customer_unique_id.transform('count')>1].copy()
        only_unique_customers['order_purchase_timestamp'] = pd.to_datetime(only_unique_customers['order_purchase_timestamp'])
        only_unique_customers['InvoiceDate'] = only_unique_customers['order_purchase_timestamp'].dt.date
        
        order_customer_payment = pd.merge(only_unique_customers, order_payment, on="order_id", how="inner")
        self.data = pd.merge(order_customer_payment, orders_reviews, on="order_id", how="inner")
        self.reference_data = self.data.loc[self.data['InvoiceDate'] < self.train_start_date]
        self.reference_data = self.reference_data.groupby('customer_unique_id').agg({'InvoiceDate': lambda x: ((self.train_end_date + dt.timedelta(days=1))- x.max()).days,"order_id": "count", "payment_value": "sum", "review_score": "mean", "review_comment_message": lambda x: np.mean(x.str.len()), "payment_installments": "mean"})
    
    def set_current_data(self, execution_date):
        self.execution_date = pd.to_datetime(execution_date).date()
        self.current_data = self.data.loc[(self.data['InvoiceDate'] < self.execution_date) & (self.data['InvoiceDate'] > self.train_end_date)]
        self.current_data = self.current_data.groupby(['customer_unique_id']).agg({'InvoiceDate': lambda x: ((self.execution_date + dt.timedelta(days=1))- x.max()).days,"order_id": "count", "payment_value": "sum", "review_score": "mean", "review_comment_message": lambda x: np.mean(x.str.len()), "payment_installments": "mean"})
    
    def get_drift_metrics(self):
        kmeans = KMeans(n_clusters=4, random_state=1).fit(self.reference_data)
        cluster_labels = kmeans.predict(self.current_data)
        self.silhouette_score = silhouette_score(self.current_data, cluster_labels)
    
    def log_metric(self):
         mlflow.set_tracking_uri("http://localhost:5000")
         experiment_name = "segmentation_maintenance"
         mlflow.set_experiment(experiment_name)
         with mlflow.start_run():
             mlflow.log_metric("silhouette_score", self.silhouette_score)
             mlflow.log_param("execution_date", str(self.execution_date))
             mlflow.log_param("train_end_date", str(self.train_end_date))
             
    def set_new_reference_data_and_retrain(self):
    if self.silhouette_score < 0.33:
        self.train_end_date = self.execution_date
        new_reference_data = self.data.loc[self.data['InvoiceDate'] < self.train_end_date]
        self.reference_data = new_reference_data.groupby('customer_unique_id').agg({
            'InvoiceDate': lambda x: (self.train_end_date - x.max()).days + 1,
            'order_id': 'count',
            'payment_value': 'sum',
            'review_score': 'mean',
            'review_comment_message': lambda x: np.mean(x.str.len()),
            'payment_installments': 'mean'
        })
        self.model = KMeans(n_clusters=self.k).fit(self.reference_data)
        
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("segmentation_maintenance")
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="sklearn-model",
                registered_model_name="k_means"
            )
        print("Model retrained and saved in MLflow")

        client = MlflowClient()
        latest_versions = client.get_latest_versions(name="k_means")
        version_models = [version.version for version in latest_versions]
        client.transition_model_version_stage(
            name="k_means",
            version=max(version_models),
            stage="production",
            archive_existing_versions=True
        )
    else:
        print("No need to retrain the model")

       

        
        

    
    
    
    
    
    
        