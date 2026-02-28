PR: PR-UI-DELETE-CONFIRM-01
Title: Simplify delete confirmation dialog (Yes/No, remove typing DELETE)

Context
The UI currently requires the user to type the literal string "DELETE" to confirm deleting a record. This is too heavy for day-to-day use. Replace it with a standard confirmation dialog with Cancel and Delete actions.

Scope
- UI: replace typed confirmation with a Yes/No modal dialog
- API: keep the existing delete behavior; if the backend currently requires a confirmation payload/token, make it optional or remove that requirement
- Tests: update any existing unit/e2e tests affected by the flow change

Requirements
1) UX
- When the user clicks Delete, show a modal dialog:
  - Title: "Delete record?"
  - Copy: "This action can’t be undone."
  - Show identifying context for the record (at minimum: display name/title if available, plus date/time or id).
- Buttons:
  - Cancel (secondary) and Delete (danger)
  - Default focus must be on Cancel
  - Escape key closes the dialog as Cancel
- While the delete request is in flight:
  - Disable the Delete button (and ideally show a small loading state)
  - Prevent double-submit

2) Safety and consistency
- Keep any existing backend protections (auth, rate limits, etc).
- Ensure the UI cannot delete the wrong item due to stale state:
  - The dialog must be bound to a specific record id.
  - After success, the UI must refresh the list/detail state and navigate appropriately.

3) Backend contract
- Find the delete endpoint and inspect whether it expects an explicit confirmation string in the request body (for example: confirm="DELETE").
- If such a requirement exists, change it to not require typed confirmation. Options:
  - Best: remove the field from the request model and accept a normal DELETE with no body.
  - Acceptable: keep the field but make it optional and ignore it.
- Ensure this change is backward compatible for existing callers if needed (for example, accept both old and new requests for 1 release).

Implementation plan
1) Locate the typed confirmation
- Search the frontend for the string DELETE and for delete-confirm components.
- Identify the component that renders the typed input and the handler that triggers the delete API call.

2) Implement the new confirmation modal
- Replace the typed input with a standard modal/confirm dialog.
- Ensure the record identifier (id) is captured at the moment the modal opens.
- Ensure Cancel is the default focused action.

3) Integrate with delete API
- Keep the existing request method (DELETE or POST) as-is unless you also adjust the backend.
- If you change the backend to a pure DELETE, update the client accordingly.

4) State updates
- After delete succeeds:
  - Show a toast/notification (if the app already uses them).
  - Invalidate/refetch the deleted record queries.
  - Navigate away from a deleted detail page if needed.

5) Update tests
- Update any tests that previously typed DELETE.
- Add a focused test for:
  - Opening the modal
  - Cancel keeps the record
  - Confirm Delete removes the record and UI updates

Validation checklist
- Manual:
  - Delete from list view works
  - Delete from detail view works (if applicable)
  - Cancel keeps the record
  - Keyboard: Escape cancels; default focus is Cancel
- Automated:
  - Unit tests pass
  - E2E tests pass
  - CI passes

Success criteria
- No UI flow requires typing "DELETE" to delete a record.
- Deleting a record always requires an explicit click on the Delete (danger) button in a modal.
- Cancel is the default action and Escape cancels.
- All relevant tests and CI checks are green.
