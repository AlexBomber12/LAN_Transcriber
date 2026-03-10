ALTER TABLE calendar_events
    ADD COLUMN organizer_name TEXT;

ALTER TABLE calendar_events
    ADD COLUMN organizer_email TEXT;

ALTER TABLE calendar_events
    ADD COLUMN attendees_json TEXT NOT NULL DEFAULT '[]';
