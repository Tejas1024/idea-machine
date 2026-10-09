"""The host PC runs every day (L16.1). On the bot host (Tejas's PC), from Task Scheduler:

    python -m machine.host daily        the daily run: prices from MT5 (sealed), caches, the ideas, the weekly
                                        questions, MLflow, then one summary to Tejas
    python -m machine.host heartbeat    the hourly heartbeat: was a beat missed (the PC off)? is the bot running?
                                        did the daily run work? an alert to Tejas for each new problem

lock.py one run at a time · alert.py messages to Tejas · power.py what Windows says about restarts · prices.py the
sealed prices · caches.py · mlflow_sync.py · daily.py · heartbeat.py · tasks.py Task Scheduler · __main__.py the commands.
"""
