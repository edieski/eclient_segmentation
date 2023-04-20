import pandas as pd
import os 
import numpy as np
import datetime as dt 
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import evidently
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class MonitoringandRetraining():
    def __init__(self):
        self.train_start_date = pd.to_datetime('2017-01-01').date()
        self.train_end_date= pd.to_datetime('2018-01-01').date()
        self.data = None
        self.reference_data = None
        self.preprocessed_data = None
        self.drift_metrics = None
    
    def load_data(self):
        project_path = os.getcwd()
        doc_path = os.path.join(project_path, "dataset")
        fichiers = os.listdir(doc_path)
        cutsomers = pd.read_csv(os.path.join(doc_path, fichiers[0]))
        orders_dataset = pd.read_csv(os.path.join(doc_path, fichiers[2]))
        orders_reviews = pd.read_csv(os.path.join(doc_path, fichiers[5]))
        customer_orders = pd.merge(orders_dataset, cutsomers, on = "customer_id", how = "inner")
        only_unique_customers = customer_orders[customer_orders.groupby('customer_unique_id').customer_unique_id.transform('count')>1].copy() 
        only_unique_customers.groupby("customer_unique_id").agg({"order_id": "count"}).sort_values(by = "order_id", ascending = False).head(100)
        def get_month(x): return dt.datetime(x.year, x.month, 1)
        only_unique_customers['order_purchase_timestamp'] = pd.to_datetime(only_unique_customers['order_purchase_timestamp'])
        only_unique_customers['InvoiceDate'] = [d.date() for d in only_unique_customers["order_purchase_timestamp"]]
        order_customer_paiement = pd.merge(only_unique_customers, order_payment,  on = "order_id", how = "inner")
        order_customer_paiement_review = pd.merge(order_customer_paiement, orders_reviews,  on = "order_id", how = "inner")
        start_date = pd.to_datetime('2017-01-01').date()
        self.end_date = pd.to_datetime('2018-01-01').date()
# filter the dataframe by date range
        self.reference_data = order_customer_paiement_review.loc[(order_customer_paiement_review['InvoiceDate'] > self.train_start_date) & (order_customer_paiement_review['InvoiceDate'] < self.train_start_dateend_date)]
    
    def get_weeks(self):

    def get_drift_metrics(self):
        DataDriftPreset(stattest='PSI', sttatest_threshold=0.2)
        data_drift_dataset_report = Report(DataDriftPreset())
        data_drift_report.run(current_data = self.reference_data, )
        data_drift_dataset_report.json()
        self.drift_metrics = data_drift_dataset_report.as_dict()
    
    def train(self):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k).fit(new_data
        mlflow.sklearn.log_model(
        sk_model=sk_learn_rfr,
        artifact_path="sklearn-model",
        registered_model_name="k_means"
        if 
        latest_versions = client.get_latest_versions(name=model_name)
        version_models = []
        for version in latest_versions:
            version_models.append(version.version)
        new_model


        
        
        
        

    
    
    
    
    
    
        