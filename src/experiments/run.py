def run_experiment(setup, train, model, evaluator, **kwargs):
    train_dataloader, test_dataloader, val_dataloader = setup(**kwargs)
    model = train(model, train_dataloader, val_dataloader, **kwargs)
    metr = evaluator.evaluate_dataloader(test_dataloader, model, **kwargs)
    
    print(metr)

    return model, metr

