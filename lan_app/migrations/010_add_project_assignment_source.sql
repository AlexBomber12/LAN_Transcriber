ALTER TABLE recordings ADD COLUMN project_assignment_source TEXT;

UPDATE recordings
SET project_assignment_source = 'manual'
WHERE project_id IS NOT NULL AND project_assignment_source IS NULL;
