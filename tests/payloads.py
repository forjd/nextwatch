"""Trimmed TMDB payloads shaped like real API responses, for normalise tests."""


def movie_payload(tmdb_id=603, **over):
    p = {
        "id": tmdb_id,
        "title": "The Matrix",
        "original_title": "The Matrix",
        "original_language": "en",
        "overview": "A hacker learns the truth.",
        "tagline": "Welcome to the Real World.",
        "release_date": "1999-03-31",
        "runtime": 136,
        "status": "Released",
        "imdb_id": "tt0133093",
        "vote_count": 26000,
        "vote_average": 8.2,
        "popularity": 80.5,
        "poster_path": "/m.jpg",
        "adult": False,
        "video": False,
        "belongs_to_collection": {"id": 2344, "name": "The Matrix Collection"},
        "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
        "production_companies": [{"id": 79, "name": "Village Roadshow", "origin_country": "US"}],
        "production_countries": [{"iso_3166_1": "US", "name": "United States"}],
        "keywords": {"keywords": [{"id": 310, "name": "artificial intelligence"}]},
        "credits": {
            "cast": [
                {"id": 2975, "name": "Laurence Fishburne", "order": 1, "character": "Morpheus",
                 "gender": 2, "known_for_department": "Acting", "popularity": 20},
                {"id": 6384, "name": "Keanu Reeves", "order": 0, "character": "Neo",
                 "gender": 2, "known_for_department": "Acting", "popularity": 40},
            ],
            "crew": [
                {"id": 9339, "name": "Lilly Wachowski", "department": "Directing",
                 "job": "Director"},
                {"id": 9339, "name": "Lilly Wachowski", "department": "Writing", "job": "Writer"},
                {"id": 1, "name": "Some Grip", "department": "Crew", "job": "Grip"},
                {"id": 2, "name": "Exec", "department": "Production", "job": "Executive Producer"},
            ],
        },
        "release_dates": {
            "results": [
                {"iso_3166_1": "GB", "release_dates": [
                    {"type": 4, "certification": "15"},
                    {"type": 3, "certification": "18"},
                ]},
                {"iso_3166_1": "US", "release_dates": [{"type": 3, "certification": "R"}]},
            ]
        },
        "watch/providers": {"results": {"GB": {
            "flatrate": [{"provider_id": 8, "provider_name": "Netflix", "display_priority": 1}],
            "rent": [{"provider_id": 2, "provider_name": "Apple TV", "display_priority": 4}],
        }}},
        "recommendations": [{"id": 604, "media_type": "movie"}, {"id": 1399, "media_type": "tv"}],
        "external_ids": {"imdb_id": "tt0133093"},
    }  # fmt: skip
    p.update(over)
    return p


def tv_payload(tmdb_id=1396, **over):
    p = {
        "id": tmdb_id,
        "name": "Breaking Bad",
        "original_name": "Breaking Bad",
        "original_language": "en",
        "overview": "A teacher turns to crime.",
        "tagline": "",
        "first_air_date": "2008-01-20",
        "episode_run_time": [45, 47, 58],
        "status": "Ended",
        "number_of_seasons": 5,
        "vote_count": 15000,
        "vote_average": 8.9,
        "popularity": 300.0,
        "poster_path": "/bb.jpg",
        "genres": [{"id": 18, "name": "Drama"}],
        "networks": [{"id": 174, "name": "AMC", "origin_country": "US"}],
        "production_companies": [{"id": 11073, "name": "Sony Pictures TV", "origin_country": "US"}],
        "production_countries": [{"iso_3166_1": "US"}],
        "created_by": [{"id": 66633, "name": "Vince Gilligan", "gender": 2}],
        "keywords": {"results": [{"id": 15484, "name": "chemistry"}]},
        "aggregate_credits": {
            "cast": [
                {"id": 84497, "name": "Aaron Paul", "order": 1, "total_episode_count": 62,
                 "roles": [{"character": "Jesse Pinkman", "episode_count": 62}]},
                {"id": 17419, "name": "Bryan Cranston", "order": 0, "total_episode_count": 62,
                 "roles": [{"character": "Walter White", "episode_count": 62}]},
                {"id": 5, "name": "Guest", "order": 50, "total_episode_count": 1,
                 "roles": [{"character": "Guest", "episode_count": 1}]},
            ],
            "crew": [
                {"id": 66633, "name": "Vince Gilligan", "department": "Writing",
                 "jobs": [{"job": "Writer", "episode_count": 13}]},
                {"id": 7, "name": "Grip", "department": "Crew", "jobs": [{"job": "Grip"}]},
            ],
        },
        "content_ratings": {"results": [{"iso_3166_1": "GB", "rating": "18"}]},
        "watch/providers": {"results": {"GB": {
            "flatrate": [{"provider_id": 8, "provider_name": "Netflix", "display_priority": 1}],
        }}},
        "recommendations": [{"id": 60059, "media_type": "tv"}],
        "external_ids": {"imdb_id": "tt0903747"},
    }  # fmt: skip
    p.update(over)
    return p
