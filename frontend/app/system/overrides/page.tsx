<div className="grid grid-cols-2 gap-4">
  <Button onClick={() => runScript("nightly")}>🛠 Run Nightly Job</Button>
  <Button onClick={() => runScript("dashboard")}>📊 Recompute Dashboard</Button>
  <Button onClick={() => runScript("insights")}>💡 Build Insights</Button>
  <Button onClick={() => runScript("train")}>🧠 Train Models</Button>
  <Button onClick={() => runScript("metrics")}>📈 Refresh Metrics</Button>
  <Button onClick={() => runScript("fundamentals")}>🏦 Fetch Fundamentals</Button>
  <Button onClick={() => runScript("news")}>📰 Update News</Button>
  <Button onClick={() => runScript("verify")}>🔍 Verify Cache</Button>
</div>
