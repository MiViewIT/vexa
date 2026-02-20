"""Add temporal workflow tracking fields to transcription_jobs

Revision ID: f9c2d8e1a4b0
Revises: a1b2c3d4e5f6
Create Date: 2026-02-19 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f9c2d8e1a4b0'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('transcription_jobs', sa.Column('workflow_id', sa.String(length=255), nullable=True))
    op.add_column('transcription_jobs', sa.Column('run_id', sa.String(length=255), nullable=True))
    op.create_index('ix_transcription_jobs_workflow_id', 'transcription_jobs', ['workflow_id'], unique=False)
    op.create_index('ix_transcription_jobs_run_id', 'transcription_jobs', ['run_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_transcription_jobs_run_id', table_name='transcription_jobs')
    op.drop_index('ix_transcription_jobs_workflow_id', table_name='transcription_jobs')
    op.drop_column('transcription_jobs', 'run_id')
    op.drop_column('transcription_jobs', 'workflow_id')
