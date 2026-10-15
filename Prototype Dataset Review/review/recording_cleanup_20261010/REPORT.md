# Recording cleanup - October 10, 2026

Reviewed every decoded frame of October 8 PASSTHRU takes 89-106 in contact sheets. Moved takes 89-104 (16 videos and matching metadata) to WAIT for walking outside the tabletop interior without observed task handling.

Takes 105-106 remain in PASSTHRU pending segment annotation: close tabletop/edge hand movement, with clear interior hand positions in take 106. Retaining the folder does not confirm every frame as PASSTHRU.

Checked all 97 recording metadata timestamps. Corrected 38 October 8 recordings to session_20261008, including filenames, take IDs, split-group IDs and internal paths. The 59 September 29 recordings already had the correct session date. Dates were taken from original timezone-aware created timestamps, not filesystem modification times.

Repaired five existing broken video references in WAIT metadata, takes 55-59: the metadata pointed to vithaal filenames while the actual matching take/recording UUID files were named jason. Updated file references only; participant identity was not reassessed.

Current source-video folder counts: INTERACTION 40, PASSTHRU 17, WAIT 40. These are not validated training segment counts.

Verified all 97 video/metadata pairs, all internal paths and session dates, and unchanged video SHA-256 hashes. No footage deleted or re-encoded. Preserved source-recording UUIDs, original capture timestamps, quality warnings and annotation confirmation status. Original metadata backups and old/new mappings are in this directory. No existing derived dataset references to the affected October 8 take IDs were found. The September 29 pilot export was not rebuilt.
