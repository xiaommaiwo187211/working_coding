一、 模型训练集选择：
1. self,一阶: dev.self_train_dataset（详细数据）   
   app.self_promo_train_data（promotion_id）

2. self,二级: dev.self_train_dataset_2_stages（详细数据） 
   app.self_promo_train_data_2_stages（promotion_id）

3. book,一阶: dev.books_train_dataset（详细数据）   
   app.books_promo_train_data（promotion_id）

4. book,二级: dev.books_train_dataset_2_stages（详细数据） 
   app.books_promo_train_data_2_stages（promotion_id）

二、 模型训练特征

1. dev.black_list_model_feature_books

2. dev.black_list_model_feature_self

三. 模型效果

1. dev.dev_black_selection_model_record_self

2. dev.dev_black_selection_model_record_book_self

四. 模型及特征表评估

1. 模型效果评估：dev.dev_black_list_model_monitor

2. 模型训练特征效果评估：dev.dev_black_list_features_monitor