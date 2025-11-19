ENABLE   = True
TIMEZONE = "America/Denver"

SCHEDULE = [
    # Nightly rebuild after close (Denver time)
    {"name":"nightly_full","time":"17:30","script":"backend/nightly_job.py","args":[],

     "description":"Full nightly rebuild: backfill, metrics, build, train, drift, insights."},

    # Insights before the next day
    {"name":"evening_insights","time":"18:00","script":"backend/insights_builder.py","args":[],

     "description":"Rebuild insights (top picks/filters) after nightly."},

    # Social sentiment (optional – only runs if PRAW keys exist)
    {"name":"social_sentiment","time":"20:30","script":"backend/social_sentiment_fetcher.py","args":[],

     "description":"Collect Reddit/FinBERT sentiment."},

    # Day-trading core (every hour during market hours)
    {"name":"dt_full_kickoff","time":"07:25","script":"dt_backend/daytrading_job.py","args":["--mode","full"],

     "description":"Pre-market full DT prep."},
    {"name":"dt_hourly_0930","time":"09:30","script":"dt_backend/daytrading_job.py","args":[],"description":"DT hourly pass"},
    {"name":"dt_hourly_1030","time":"10:30","script":"dt_backend/daytrading_job.py","args":[],"description":"DT hourly pass"},
    {"name":"dt_hourly_1130","time":"11:30","script":"dt_backend/daytrading_job.py","args":[],"description":"DT hourly pass"},
    {"name":"dt_hourly_1230","time":"12:30","script":"dt_backend/daytrading_job.py","args":[],"description":"DT hourly pass"},
    {"name":"dt_hourly_1330","time":"13:30","script":"dt_backend/daytrading_job.py","args":[],"description":"DT hourly pass"},
    {"name":"dt_hourly_1430","time":"14:30","script":"dt_backend/daytrading_job.py","args":[],"description":"DT hourly pass"},

    # Nightly bots (pre-market full) across horizons
    {"name":"night_bot_full_run_1w","time":"07:00","script":"backend/trading_bot_nightly_1w.py","args":["--mode","full"],"description":"1w bots full pre-market"},
    {"name":"night_bot_full_run_2w","time":"07:00","script":"backend/trading_bot_nightly_2w.py","args":["--mode","full"],"description":"2w bots full pre-market"},
    {"name":"night_bot_full_run_4w","time":"07:00","script":"backend/trading_bot_nightly_4w.py","args":["--mode","full"],"description":"4w bots full pre-market"},
]
