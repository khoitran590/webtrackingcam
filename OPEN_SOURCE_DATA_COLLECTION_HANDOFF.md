# Open source and tester data collection handoff

Date: 2026-10-15

Status: Planning document for implementation agents. Creating this document does not authorize provisioning services, publishing the repository, uploading recordings, or contacting testers.

## 1. Objective and scope

Prepare this hand gesture and facial expression tracker for open source release and allow testers to contribute selected recordings through several sharing methods.

The owner wants to collect useful hand and face movement samples while giving testers control over what they record, keep, export, and send. Public source code does not require public access to collected samples.

This document combines the project scan and the subsequent discussion. Product requirements below describe the intended experience. Provider selection, schema details, and delivery phases are implementation proposals, not already implemented features or finalized owner decisions.

### Intended product requirements

- Testers can keep recordings locally without sending them.
- Testers can export selected recordings as a portable ZIP and send it through a channel they choose.
- Testers can review and directly upload selected recordings to the owner's private storage.
- Testers can choose which data categories to include.
- Testers can select individual takes; selecting a time range within a take is a later enhancement.
- Video recording and sharing require a separate explicit choice.
- Permission to contribute privately for model improvement is separate from permission to publish samples publicly.

### Outside the initial implementation

- Automatically uploading all webcam activity.
- Publishing collected recordings as an open dataset.
- Training a dynamic gesture or facial movement sequence model.
- Building a full web application or replacing the existing desktop app without a separate requirement.
- Automatically sending exports by email, chat, or other messaging services.

## 2. Current project baseline

Recheck the working tree before implementation. There are existing uncommitted changes in README.md, SPEC_NOTES.md, tests/test_sample_images.py, tools/fit_expression_offsets.py, and tools/record_samples.py at the time of this handoff. Preserve them and coordinate before editing these files.

| Area | Observed implementation |
| --- | --- |
| Runtime | Local Python webcam application using MediaPipe Tasks and OpenCV |
| Hand processing | Gesture rules, learned classifier, temporal smoothing, and tracking in hand_gesture_tracker.py, gesture_logic.py, and gesture_model.py |
| Face processing | Landmark detection, expression scores, smoothing, face tracking, and hand occlusion handling in face_expression.py |
| Hand recorder | tools/record_samples.py saves labeled hand takes as JSONL under samples/ |
| Hand record fields | class, image_label, size, points_px, and world |
| Face recorder | The current working tree also records expression class and network scores under samples/faces/ |
| Face sample use | tools/fit_expression_offsets.py can use personal expression takes to fit expression offsets |
| Movement limitations | Existing sample rows lack per-frame timestamps, frame numbers, and track IDs; face sample files do not contain facial landmark trajectories |
| Face landmark exposure | FacePipeline computes facial points internally, but FaceReading does not expose the full landmark array |
| Training | tools/train_gesture_model.py consumes existing hand takes; this is static gesture classification |
| Cloud capabilities | No shared database, upload service, contributor authentication, or cloud sample storage found |
| Repository hygiene | .gitignore excludes samples/, downloaded model binaries, and dataset caches |
| Licensing | No top-level project LICENSE file found |

Important correction to the original scan: face expression score recording now exists in the working tree. Do not implement it again under the assumption that all face recording is missing. Saving expression scores is still different from saving face movement coordinates.

## 3. Tester experience

Use the existing desktop context when choosing controls. Do not assume a browser interface is required. Keyboard commands, a small desktop review window, or a companion command can be considered, provided the choices remain clear and accessible.

### Before recording

1. Explain what will be recorded and that local recording does not send data.
2. Let the tester enable hand movement, face movement, or both.
3. Offer recognition results and optional feedback as separate inclusions.
4. Keep video disabled by default. Explain that video can contain faces, surroundings, and other people.
5. Start only after an explicit recording action and show a visible recording indicator.
6. Allow stopping or cancelling a take. Define how incomplete takes are handled.

### After recording

Present the following actions without forcing a contribution:

| Action | Required behavior |
| --- | --- |
| Keep locally | Preserve the recording on the tester's device; no upload request is made |
| Export ZIP | Export only selected takes and enabled data categories; the tester chooses where to save it |
| Upload selected | Show a review, obtain contribution permission, and upload selected data to private storage |
| Delete local recording | Remove the selected local take after a clear deliberate action |

Use the same validated package format for export and upload. The tester should not need an account just to record locally or export. Whether direct upload requires an account remains an owner decision.

### Review before export or upload

Show selected takes, total duration, included data categories, whether video is present, approximate package size, and optional notes. Before upload also show the recipient, access scope, intended use, and the applicable consent text.

The tester can return to selection, remove takes, change included categories, or cancel. Cancelling must not upload or delete the original local takes. Exporting must not automatically send a message or publish a link.

### Later enhancement: share a time range

Allow trimming the exported or uploaded copy to a selected interval. Apply the interval to every included stream, including video. Regenerate duration, timestamps, and counts consistently. Preserve the original local take.

## 4. Data categories and defaults

| Category | Proposed contents | Default behavior |
| --- | --- | --- |
| Hand movement | Landmark coordinates, handedness, relative timing, and session-scoped track IDs | Included only when hand recording is enabled |
| Face movement | Facial landmark coordinates, relative timing, and session-scoped track IDs | Included only when face recording is enabled |
| Recognition results | Predicted labels, confidence, class probabilities or scores, smoothing state, and occlusion flags where available | Clearly selectable; never substituted for human labels |
| Human labels | Prompted gesture/expression and any tester correction, with label source | Kept separate from model predictions |
| Feedback | Optional notes about errors, lighting, camera placement, or expected behavior | Empty unless the tester supplies it |
| Video | Synchronized video of the selected takes | Off; requires explicit enablement and preview |

Face and hand coordinates should not be described as anonymous merely because the image is absent. Avoid collecting real names, email addresses, machine usernames, absolute file paths, or stable hardware identifiers in sample files by default. Free-text notes can contain personal information; make them reviewable.

Do not silently retain excluded predictions, facial coordinates, or video in metadata, diagnostic logs, exports, or uploaded archives. If an optional stream is absent, explain which evaluation or training uses become unavailable.

## 5. Recording format proposal

Store one versioned package per contribution, with recordings grouped into takes. Start with compressed JSONL for frame data and a JSON manifest. Consider another format only if measured volume justifies it.

Suggested package structure:

```text
contribution.zip
  manifest.json
  takes/
    <take-id>/
      frames.jsonl
      video.mp4       # present only when explicitly included
```

Suggested manifest fields:

- schema_version, package_id, session_id, and take IDs.
- App version or commit and detector/classifier versions or hashes.
- Included data categories and an inventory of files with sizes and checksums.
- Frame dimensions, mirroring, landmark topology/version, coordinate units, axes, and reference frames.
- Measured duration and sample count; requested capture rate if relevant.
- Human label and its source, plus optional tester notes.
- Permission record reference and consent-text version when contributing; local-only exports can explicitly have no contribution grant.

Suggested frame fields:

- frame_index and monotonic elapsed time relative to the take start.
- Explicit capture/detection status so missing detections and dropped frames are distinguishable.
- Separate hand and face arrays, each with a session-scoped track ID.
- Available raw landmarks and, if needed, explicitly marked smoothed landmarks.
- Image and hand world coordinates in separately documented fields. Do not treat MediaPipe hand world coordinates as a shared global scene coordinate system.
- Face landmark coordinates with documented depth semantics if depth is retained. Do not invent physical units for detector-relative depth.
- Optional predictions, raw versus smoothed scores, confidence, and coverage/occlusion status.

All streams from a captured frame must share its timestamp. Do not infer timing solely from row order or nominal FPS. Multiple hands/faces belong to the same frame record rather than being mistaken for successive frames. Track IDs indicate temporary tracking continuity, not verified identity.

Keep class order, score transformation, and expression offset version when preserving face scores. Current personal face samples store scores with the app's offsets removed; future consumers must not confuse these with final displayed probabilities.

Preserve old hand and face sample readers, or provide explicit adapters/migrations. Never fabricate missing timestamps for legacy recordings or present them as complete movement trajectories.

## 6. Storage and contribution architecture proposal

Recommended starting option: Supabase Postgres for metadata plus a private Supabase Storage bucket for sample packages. This is a proposal; no service has been provisioned or provider selected by the owner.

| Layer | Responsibility |
| --- | --- |
| Local recorder | Capture selected streams, save locally, and expose review/export controls |
| Package builder | Filter takes/categories, build manifests, validate data, and create export/upload packages |
| Contribution endpoint | Authorize uploads, validate requests and quotas, and issue narrowly scoped upload access |
| Private object storage | Store immutable sample packages; no public bucket listing or public sample URLs |
| Metadata database | Track contributions, permissions, file references, validation, retention, and deletion |
| Owner ingestion tools | Validate contributed packages and prepare approved samples for evaluation or training |

Suggested database entities: contributors (pseudonymous where possible), sessions, takes, contributions, consent_records, and sample_files. Store dense frame arrays in files initially rather than creating a database row per landmark.

### Upload flow

1. Tester reviews and explicitly selects Upload.
2. Client builds and validates the same package used by ZIP export.
3. Service validates authorization, allowed size/types, intended data categories, and contribution permission.
4. Client uploads using restricted access; administrative credentials stay on the server.
5. Service verifies the package and marks it complete only after file and metadata checks pass.
6. Tester receives a receipt/contribution ID and instructions for requesting deletion.

Handle offline operation, cancellation, interrupted uploads, retries, duplicate submissions, and incomplete objects. Use an idempotency mechanism so retrying does not create duplicate accepted contributions. Preserve local recordings if upload fails.

If account-free uploads are supported, define scoped submission credentials, limits, and a secure deletion receipt. Do not ship a shared unrestricted write key. Keep backend configuration outside source control and document setup with placeholder values.

### Manual contributions

ZIP export must work offline. The owner can import a manually received ZIP through the same validation path as a direct upload. Do not automatically ingest packages from third-party messaging services. Validate archive paths, expansion size, schema, file inventory, numeric values, and declared categories before accepting a package.

## 7. Consent, access, and lifecycle

- Separate local recording, private submission for stated purposes, and public dataset publication.
- Do not bundle permission for public publication into the Upload action.
- Record which consent wording and data categories the tester approved, when, and for which contribution.
- Provide a deletion request mechanism without requiring personal details in each frame file.
- Define retention duration, who can access data, and how expired or withdrawn contributions are excluded from future datasets.
- Document deletion behavior for originals, derived datasets, backups, and already trained models. Do not promise that deleting a source recording reverses previous model training.
- Public release of any samples requires a separate dataset policy, permission review, and dataset license decision.
- Define the initial eligible tester population before inviting contributors; do not silently assume permission to collect recordings from minors or bystanders.

These are product and implementation requirements, not a legal clearance determination. Resolve applicable participant and dataset terms before collection or publication.

## 8. Open source readiness work

1. Confirm ownership and permission to publish all included code and contributions.
2. Choose a project code license. MIT and Apache-2.0 were discussed as options; neither has been selected.
3. Add a third-party inventory covering dependencies, downloaded models, test assets, training data, and derived artifacts.
4. Review the committed HaGRID-trained classifier's distribution terms separately from original application code.
5. Review MediaPipe assets, the EmotiEffLib expression model, and any required notices. Do not assume a library's code license automatically clears every model or training dataset.
6. Review RAF-DB/AffectNet access terms and dataset-derived expression offsets. Evaluation scripts currently use unofficial mirrors; a mirror's availability does not establish permission.
7. Keep downloaded third-party images, personal samples, credentials, and private storage details out of the repository. Scan Git history as well as current files for secrets and private data.
8. Document setup, contribution format, recording behavior, privacy choices, and backend configuration with safe placeholders.
9. Provide synthetic fixtures for examples and automated checks so contributors need not share personal recordings to develop the feature.

Reference starting points from the discussion (verify current terms during implementation):

- [GitHub repository licensing](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- [HaGRID license entry point](https://github.com/hukenovs/hagrid#license)
- [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)
- [AffectNet publisher information](https://www.mohammadmahoor.com/pages/databases/affectnet/)
- [RAF-DB publisher information](http://www.whdeng.cn/RAF/model1.html)
- [Supabase private storage buckets](https://supabase.com/docs/guides/storage/buckets/fundamentals)
- [Supabase storage access policies](https://supabase.com/docs/guides/storage/security/access-control)

## 9. Delivery phases and agent handoffs

### Phase 1: Recording format and local export

Deliver the shared schema, timestamps and tracking metadata, facial landmark exposure, local recording controls, take/category selection, review, and portable ZIP export. Preserve existing training and personal face offset fitting. No cloud dependency is needed to finish this phase.

Acceptance criteria:

- Hand-only, face-only, and combined takes have documented synchronized timing.
- Missing detections and multiple tracked objects are represented correctly.
- Local recording and ZIP export work without contribution accounts or network access once required inference models are available locally.
- Exported archives contain only selected takes/categories; video is absent by default.
- Existing hand and face samples remain readable through supported paths.
- Human labels are distinguishable from predictions.

### Phase 2: Private contribution service

Deliver upload authorization, private storage policies, metadata/consent records, retry handling, contribution receipts, and deletion handling. Require the owner's provider and operating-policy decisions before provisioning dependent infrastructure.

Acceptance criteria:

- Only explicitly submitted packages are transmitted.
- Ordinary contributors cannot list or read other contributors' files.
- No administrative secrets are present in the desktop app or repository.
- Cancelled/failed uploads preserve local recordings and do not appear as completed contributions.
- Repeated requests cannot create duplicate accepted contributions.
- Deletion and retention behavior match the documented policy.

### Phase 3: Owner ingestion and evaluation

Deliver shared validation for ZIP imports and direct uploads, schema-aware readers, provenance tracking, and training/evaluation adapters. Group evaluation splits by contributor/session where feasible so adjacent frames or repeated takes do not leak into held-out results. Do not claim dynamic movement recognition from merely storing sequences.

### Phase 4: Additional sharing controls

Deliver time-range selection and, if approved, optional synchronized video. Ensure excluded time ranges are physically absent from delivered files, including video. Keep public dataset publishing as a separate project decision.

### Suggested work packages

| Agent work package | Outputs and dependencies |
| --- | --- |
| Schema and recorder | Versioned schema, compatibility adapters, synchronized capture; establishes contracts first |
| Tester controls and export | Review flow and package builder; depends on schema |
| Contribution backend | Upload service, policies, metadata, receipts, deletion; depends on package contract and owner decisions |
| Ingestion and validation | Shared validators and owner import tools; coordinates with package/backend contracts |
| Release readiness | Ownership/license inventory, secret scan, setup/privacy/contribution documentation |

Implementation agents should agree on schema and module ownership before parallel edits. Existing changes to the recorder and expression offset fitter must be retained and integrated.

## 10. Decisions still needed

These should not prevent schema design or local export work, but dependent features must not silently choose policies:

- Final code license and clearance for distributed model artifacts.
- Storage provider/account and hosting configuration.
- Whether direct upload requires accounts, invitation tokens, or another scoped mechanism.
- Exact contribution purposes: debugging, evaluation, training, and/or other uses.
- Whether video is needed, and its size/duration limits.
- Retention duration, deletion process, backup handling, and eligible tester population.
- Whether any future public dataset will be offered and under what separate permission and terms.

## 11. Verification and completion report

Use synthetic recordings for new validation and sharing tests. Verify meaningful behavior: stream exclusion, timing, legacy compatibility, archive validation, failed/retried uploads, access isolation, and deletion. Run relevant existing hand/face tests when those paths change. Do not record or upload personal webcam data as part of automated checks.

At handoff completion, report files changed, implemented phases, validation performed, known limitations, owner decisions still pending, and any infrastructure or publication actions actually taken. Do not describe proposed features as shipped.
