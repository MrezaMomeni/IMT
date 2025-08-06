import runpy

if __name__ == "__main__":
    runpy.run_path('src/joint_models/RandomForest_model_soheib_khaledian.py')
    runpy.run_path('src/joint_models/XGbost_model_soheib_khaledian.py')

    runpy.run_path('src/single_target_models/expenses/RandomForest_model_soheib_khaledian.py')
    runpy.run_path('src/single_target_models/expenses/XGbost_model_soheib_khaledian.py')

    runpy.run_path('src/single_target_models/premium/RandomForest_model_soheib_khaledian.py')
    runpy.run_path('src/single_target_models/premium/XGbost_model_soheib_khaledian.py')