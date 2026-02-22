from src.core.lastfm_client import LastFMClient

def test_lastfm_connection():
    print("🔍 Testing Last.fm API Connection...")
    
    try:
        # 1. Initialize the client
        client = LastFMClient()
        
        # 2. Fetch info for a global artist to verify signal quality
        artist_to_test = "Queen"
        print(f"📡 Requesting data for: {artist_to_test}...")
        
        data = client.get_artist_info(artist_to_test)
        
        # 3. Validation
        if data["listeners"] > 0:
            print("✨ Success! Connected to Last.fm API.")
            print(f"👤 Artist: {artist_to_test}")
            print(f"🎧 Listeners: {data['listeners']:,}")
            print(f"🏷️ Top Tags: {', '.join(data['tags'][:5])}")
            
            if not data["tags"]:
                print("⚠️ Warning: Connected, but no tags were found.")
        else:
            print("❌ Connection failed or artist not found.")
            print(f"📝 Response Data: {data}")

    except Exception as e:
        print(f"💥 An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_lastfm_connection()