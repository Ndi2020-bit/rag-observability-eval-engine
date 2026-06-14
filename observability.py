import phoenix as px
import time

print("🚀 Launching localized Arize Phoenix Tracing Server...")

# Force the application instance to spin up globally
session = px.launch_app()

print(f"\n🎯 Dashboard is live! View it here: {session.url}")
print("Keep this terminal open to log incoming RAG traces.")

# Create a robust, explicit keep-alive loop
try:
    while True:
        time.sleep(2)
except KeyboardInterrupt:
    print("\n🛑 Shutting down tracking engine...")
